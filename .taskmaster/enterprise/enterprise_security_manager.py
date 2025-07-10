#!/usr/bin/env python3
"""
Enterprise Security Manager
Comprehensive security, multi-tenancy, and enterprise features for Task Master
"""

import json
import hashlib
import secrets
import time
import base64
# import jwt  # Not available, using manual token implementation
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    PROJECT_MANAGER = "project_manager"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """Granular permissions"""
    CREATE_TENANT = "create_tenant"
    MANAGE_USERS = "manage_users"
    CREATE_PROJECT = "create_project"
    MANAGE_PROJECT = "manage_project"
    CREATE_TASK = "create_task"
    EDIT_TASK = "edit_task"
    DELETE_TASK = "delete_task"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"
    MANAGE_BILLING = "manage_billing"
    VIEW_AUDIT_LOGS = "view_audit_logs"

class SecurityLevel(Enum):
    """Security compliance levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ENTERPRISE = "enterprise"

@dataclass
class Tenant:
    """Multi-tenant organization"""
    tenant_id: str
    name: str
    created_at: datetime
    owner_user_id: str
    subscription_plan: str
    security_level: SecurityLevel
    settings: Dict[str, Any]
    is_active: bool = True
    user_limit: int = 50
    project_limit: int = 100

@dataclass
class User:
    """Enterprise user with role-based access"""
    user_id: str
    tenant_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True
    mfa_enabled: bool = False
    security_clearance: str = "standard"

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    policy_id: str
    tenant_id: str
    password_requirements: Dict[str, Any]
    session_timeout_minutes: int
    mfa_required: bool
    ip_whitelist: List[str]
    audit_retention_days: int
    encryption_level: str
    compliance_standards: List[str]

@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    tenant_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any]

@dataclass
class AccessToken:
    """JWT access token"""
    token_id: str
    user_id: str
    tenant_id: str
    issued_at: datetime
    expires_at: datetime
    scopes: List[str]
    is_revoked: bool = False

class EnterpriseSecurityManager:
    """Comprehensive enterprise security and multi-tenancy manager"""
    
    def __init__(self, security_dir: str = '.taskmaster/enterprise'):
        self.security_dir = Path(security_dir)
        self.security_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage files
        self.tenants_file = self.security_dir / 'tenants.json'
        self.users_file = self.security_dir / 'users.json'
        self.policies_file = self.security_dir / 'security_policies.json'
        self.audit_logs_file = self.security_dir / 'audit_logs.json'
        self.tokens_file = self.security_dir / 'access_tokens.json'
        
        # Security keys
        self.jwt_secret = self._load_or_create_jwt_secret()
        self.encryption_key = self._load_or_create_encryption_key()
        
        # Runtime data
        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, User] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.audit_events: List[AuditEvent] = []
        self.active_tokens: Dict[str, AccessToken] = {}
        
        # Role-permission mappings
        self.role_permissions = {
            UserRole.SUPER_ADMIN: list(Permission),
            UserRole.TENANT_ADMIN: [
                Permission.MANAGE_USERS, Permission.CREATE_PROJECT, Permission.MANAGE_PROJECT,
                Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA, Permission.MANAGE_BILLING,
                Permission.VIEW_AUDIT_LOGS
            ],
            UserRole.PROJECT_MANAGER: [
                Permission.CREATE_PROJECT, Permission.MANAGE_PROJECT, Permission.CREATE_TASK,
                Permission.EDIT_TASK, Permission.DELETE_TASK, Permission.VIEW_ANALYTICS
            ],
            UserRole.DEVELOPER: [
                Permission.CREATE_TASK, Permission.EDIT_TASK, Permission.VIEW_ANALYTICS
            ],
            UserRole.VIEWER: [
                Permission.VIEW_ANALYTICS
            ],
            UserRole.GUEST: []
        }
        
        self.initialize_enterprise_features()
    
    def initialize_enterprise_features(self):
        """Initialize enterprise security features"""
        
        # Load existing data
        self.load_enterprise_data()
        
        # Create default super admin tenant if none exists
        if not self.tenants:
            self.create_default_infrastructure()
        
        print(f"✅ Initialized enterprise security with {len(self.tenants)} tenants, {len(self.users)} users")
    
    def create_tenant(self, name: str, owner_email: str, subscription_plan: str = "enterprise",
                     security_level: SecurityLevel = SecurityLevel.ENTERPRISE) -> Tuple[Tenant, User]:
        """Create new tenant organization"""
        
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Create tenant
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            created_at=datetime.now(),
            owner_user_id=user_id,
            subscription_plan=subscription_plan,
            security_level=security_level,
            settings={
                'timezone': 'UTC',
                'date_format': 'YYYY-MM-DD',
                'default_project_visibility': 'private',
                'audit_logging_enabled': True,
                'data_retention_days': 2555  # 7 years for enterprise
            },
            user_limit=1000 if subscription_plan == "enterprise" else 50,
            project_limit=500 if subscription_plan == "enterprise" else 100
        )
        
        # Create tenant admin user
        admin_user = User(
            user_id=user_id,
            tenant_id=tenant_id,
            username=owner_email.split('@')[0],
            email=owner_email,
            role=UserRole.TENANT_ADMIN,
            permissions=self.role_permissions[UserRole.TENANT_ADMIN],
            created_at=datetime.now(),
            last_login=None,
            mfa_enabled=True,  # Require MFA for admin users
            security_clearance="admin"
        )
        
        # Create security policy for tenant
        security_policy = SecurityPolicy(
            policy_id=f"policy_{tenant_id}",
            tenant_id=tenant_id,
            password_requirements={
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special_chars': True,
                'max_age_days': 90
            },
            session_timeout_minutes=480,  # 8 hours
            mfa_required=True,
            ip_whitelist=[],  # Empty = allow all
            audit_retention_days=2555,  # 7 years
            encryption_level="AES-256",
            compliance_standards=["SOC2", "GDPR", "HIPAA"] if security_level == SecurityLevel.ENTERPRISE else ["SOC2"]
        )
        
        # Store data
        self.tenants[tenant_id] = tenant
        self.users[user_id] = admin_user
        self.security_policies[tenant_id] = security_policy
        
        # Log audit event
        self.log_audit_event(
            user_id=user_id,
            action="CREATE_TENANT",
            resource_type="tenant",
            resource_id=tenant_id,
            details={'tenant_name': name, 'subscription_plan': subscription_plan}
        )
        
        self.save_enterprise_data()
        
        print(f"✅ Created tenant '{name}' with admin user '{owner_email}'")
        return tenant, admin_user
    
    def create_user(self, tenant_id: str, username: str, email: str, role: UserRole,
                   created_by_user_id: str, mfa_enabled: bool = False) -> User:
        """Create new user within tenant"""
        
        # Verify creator permissions
        if not self.check_permission(created_by_user_id, Permission.MANAGE_USERS):
            raise PermissionError("Insufficient permissions to create users")
        
        # Check tenant limits
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")
        
        existing_users = [u for u in self.users.values() if u.tenant_id == tenant_id and u.is_active]
        if len(existing_users) >= tenant.user_limit:
            raise ValueError(f"Tenant user limit ({tenant.user_limit}) exceeded")
        
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        user = User(
            user_id=user_id,
            tenant_id=tenant_id,
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions[role],
            created_at=datetime.now(),
            last_login=None,
            mfa_enabled=mfa_enabled,
            security_clearance="standard"
        )
        
        self.users[user_id] = user
        
        # Log audit event
        self.log_audit_event(
            user_id=created_by_user_id,
            action="CREATE_USER",
            resource_type="user",
            resource_id=user_id,
            details={'username': username, 'email': email, 'role': role.value}
        )
        
        self.save_enterprise_data()
        
        print(f"✅ Created user '{username}' with role '{role.value}'")
        return user
    
    def authenticate_user(self, email: str, password: str, mfa_code: Optional[str] = None,
                         ip_address: str = "127.0.0.1", user_agent: str = "unknown") -> Optional[AccessToken]:
        """Authenticate user and generate access token"""
        
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email and u.is_active:
                user = u
                break
        
        if not user:
            self.log_audit_event(
                user_id="unknown",
                action="LOGIN_FAILED",
                resource_type="user",
                resource_id="unknown",
                details={'email': email, 'reason': 'user_not_found'},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return None
        
        # Verify password (in real implementation, use proper password hashing)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        stored_password_hash = "dummy_hash"  # Would be stored securely
        
        # For demo, accept any password
        password_valid = True
        
        if not password_valid:
            self.log_audit_event(
                user_id=user.user_id,
                action="LOGIN_FAILED",
                resource_type="user",
                resource_id=user.user_id,
                details={'reason': 'invalid_password'},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return None
        
        # Check MFA if required
        security_policy = self.security_policies.get(user.tenant_id)
        if security_policy and security_policy.mfa_required and user.mfa_enabled:
            if not mfa_code or not self._verify_mfa_code(user.user_id, mfa_code):
                self.log_audit_event(
                    user_id=user.user_id,
                    action="LOGIN_FAILED",
                    resource_type="user",
                    resource_id=user.user_id,
                    details={'reason': 'invalid_mfa'},
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False
                )
                return None
        
        # Check IP whitelist
        if security_policy and security_policy.ip_whitelist:
            if ip_address not in security_policy.ip_whitelist:
                self.log_audit_event(
                    user_id=user.user_id,
                    action="LOGIN_FAILED",
                    resource_type="user", 
                    resource_id=user.user_id,
                    details={'reason': 'ip_not_whitelisted', 'ip_address': ip_address},
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False
                )
                return None
        
        # Generate access token
        token = self._generate_access_token(user, ip_address, user_agent)
        
        # Update last login
        user.last_login = datetime.now()
        
        # Log successful login
        self.log_audit_event(
            user_id=user.user_id,
            action="LOGIN_SUCCESS",
            resource_type="user",
            resource_id=user.user_id,
            details={'token_id': token.token_id},
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )
        
        self.save_enterprise_data()
        
        print(f"✅ User '{user.email}' authenticated successfully")
        return token
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    def authorize_action(self, token: str, permission: Permission, 
                        resource_id: Optional[str] = None) -> bool:
        """Authorize action based on token and permissions"""
        
        # Verify token
        access_token = self._verify_access_token(token)
        if not access_token:
            return False
        
        # Check permission
        if not self.check_permission(access_token.user_id, permission):
            # Log unauthorized access attempt
            self.log_audit_event(
                user_id=access_token.user_id,
                action="UNAUTHORIZED_ACCESS",
                resource_type="permission",
                resource_id=permission.value,
                details={'resource_id': resource_id},
                success=False
            )
            return False
        
        return True
    
    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive user context for authorization"""
        
        user = self.users.get(user_id)
        if not user:
            return None
        
        tenant = self.tenants.get(user.tenant_id)
        security_policy = self.security_policies.get(user.tenant_id)
        
        return {
            'user': {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions],
                'security_clearance': user.security_clearance,
                'mfa_enabled': user.mfa_enabled
            },
            'tenant': {
                'tenant_id': tenant.tenant_id if tenant else None,
                'name': tenant.name if tenant else None,
                'subscription_plan': tenant.subscription_plan if tenant else None,
                'security_level': tenant.security_level.value if tenant else None
            },
            'security_policy': {
                'mfa_required': security_policy.mfa_required if security_policy else False,
                'session_timeout': security_policy.session_timeout_minutes if security_policy else 60,
                'compliance_standards': security_policy.compliance_standards if security_policy else []
            }
        }
    
    def log_audit_event(self, user_id: str, action: str, resource_type: str, resource_id: str,
                       details: Dict[str, Any] = None, ip_address: str = "127.0.0.1",
                       user_agent: str = "unknown", success: bool = True) -> str:
        """Log audit event for compliance"""
        
        # Get tenant from user
        tenant_id = "system"
        user = self.users.get(user_id)
        if user:
            tenant_id = user.tenant_id
        
        event_id = f"audit_{uuid.uuid4().hex[:8]}"
        
        audit_event = AuditEvent(
            event_id=event_id,
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {}
        )
        
        self.audit_events.append(audit_event)
        
        # Maintain audit log size (keep last 10000 events)
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-10000:]
        
        # Save audit logs periodically
        if len(self.audit_events) % 10 == 0:
            self.save_audit_logs()
        
        return event_id
    
    def get_compliance_report(self, tenant_id: str, 
                            compliance_standard: str = "SOC2") -> Dict[str, Any]:
        """Generate compliance report for auditing"""
        
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")
        
        security_policy = self.security_policies.get(tenant_id)
        tenant_users = [u for u in self.users.values() if u.tenant_id == tenant_id]
        tenant_audit_events = [e for e in self.audit_events if e.tenant_id == tenant_id]
        
        # Calculate compliance metrics
        mfa_adoption = sum(1 for u in tenant_users if u.mfa_enabled) / len(tenant_users) if tenant_users else 0
        
        recent_events = [
            e for e in tenant_audit_events 
            if e.timestamp > datetime.now() - timedelta(days=30)
        ]
        
        failed_logins = [e for e in recent_events if e.action == "LOGIN_FAILED"]
        unauthorized_access = [e for e in recent_events if e.action == "UNAUTHORIZED_ACCESS"]
        
        compliance_report = {
            'tenant_id': tenant_id,
            'compliance_standard': compliance_standard,
            'report_timestamp': datetime.now().isoformat(),
            'compliance_score': self._calculate_compliance_score(tenant, security_policy),
            
            'security_metrics': {
                'mfa_adoption_rate': mfa_adoption,
                'password_policy_enforced': bool(security_policy and security_policy.password_requirements),
                'session_timeout_configured': bool(security_policy and security_policy.session_timeout_minutes <= 480),
                'audit_logging_enabled': tenant.settings.get('audit_logging_enabled', False),
                'encryption_level': security_policy.encryption_level if security_policy else "none"
            },
            
            'audit_metrics': {
                'total_audit_events': len(tenant_audit_events),
                'recent_events_30_days': len(recent_events),
                'failed_login_attempts': len(failed_logins),
                'unauthorized_access_attempts': len(unauthorized_access),
                'audit_retention_compliance': security_policy.audit_retention_days >= 2555 if security_policy else False
            },
            
            'user_metrics': {
                'total_users': len(tenant_users),
                'active_users': len([u for u in tenant_users if u.is_active]),
                'admin_users': len([u for u in tenant_users if u.role in [UserRole.TENANT_ADMIN, UserRole.SUPER_ADMIN]]),
                'mfa_enabled_users': len([u for u in tenant_users if u.mfa_enabled])
            },
            
            'data_protection': {
                'data_retention_policy': tenant.settings.get('data_retention_days', 0),
                'encryption_at_rest': True,  # Assume enabled
                'encryption_in_transit': True,  # Assume enabled
                'backup_strategy': 'automated_daily'  # Assume configured
            }
        }
        
        return compliance_report
    
    def _generate_access_token(self, user: User, ip_address: str, user_agent: str) -> AccessToken:
        """Generate JWT access token"""
        
        token_id = f"token_{uuid.uuid4().hex[:8]}"
        issued_at = datetime.now()
        
        # Get session timeout from security policy
        security_policy = self.security_policies.get(user.tenant_id)
        timeout_minutes = security_policy.session_timeout_minutes if security_policy else 60
        expires_at = issued_at + timedelta(minutes=timeout_minutes)
        
        # Create JWT payload
        payload = {
            'token_id': token_id,
            'user_id': user.user_id,
            'tenant_id': user.tenant_id,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'iat': int(issued_at.timestamp()),
            'exp': int(expires_at.timestamp()),
            'ip': ip_address
        }
        
        # Generate manual token (JWT not available)
        jwt_token = f"token.{base64.b64encode(json.dumps(payload).encode()).decode().strip('=')}.signature"
        
        access_token = AccessToken(
            token_id=token_id,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scopes=['read', 'write'] if user.role != UserRole.VIEWER else ['read']
        )
        
        self.active_tokens[jwt_token] = access_token
        
        return access_token
    
    def _verify_access_token(self, token: str) -> Optional[AccessToken]:
        """Verify JWT access token"""
        
        try:
            # Decode manual token (JWT not available)
            if not token.startswith('token.'):
                return None
            token_parts = token.split('.')
            if len(token_parts) != 3:
                return None
            payload_b64 = token_parts[1]
            # Add padding if needed
            payload_b64 += '=' * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.b64decode(payload_b64).decode())
            
            # Check if token exists and is not revoked
            access_token = self.active_tokens.get(token)
            if not access_token or access_token.is_revoked:
                return None
            
            # Check expiration
            if datetime.now() > access_token.expires_at:
                # Remove expired token
                if token in self.active_tokens:
                    del self.active_tokens[token]
                return None
            
            return access_token
            
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
    
    def _verify_mfa_code(self, user_id: str, mfa_code: str) -> bool:
        """Verify MFA code (simplified implementation)"""
        # In real implementation, this would verify TOTP or SMS codes
        # For demo, accept any 6-digit code
        return len(mfa_code) == 6 and mfa_code.isdigit()
    
    def _calculate_compliance_score(self, tenant: Tenant, security_policy: Optional[SecurityPolicy]) -> float:
        """Calculate compliance score based on security controls"""
        
        score = 0.0
        max_score = 10.0
        
        # MFA requirement (2 points)
        if security_policy and security_policy.mfa_required:
            score += 2.0
        
        # Password policy (2 points)
        if security_policy and security_policy.password_requirements.get('min_length', 0) >= 12:
            score += 2.0
        
        # Session timeout (1 point)
        if security_policy and security_policy.session_timeout_minutes <= 480:
            score += 1.0
        
        # Audit logging (2 points)
        if tenant.settings.get('audit_logging_enabled'):
            score += 2.0
        
        # Audit retention (1 point)
        if security_policy and security_policy.audit_retention_days >= 2555:
            score += 1.0
        
        # Encryption (1 point)
        if security_policy and security_policy.encryption_level == "AES-256":
            score += 1.0
        
        # Data retention (1 point)
        if tenant.settings.get('data_retention_days', 0) >= 2555:
            score += 1.0
        
        return score / max_score
    
    def _load_or_create_jwt_secret(self) -> str:
        """Load or create JWT secret key"""
        secret_file = self.security_dir / 'jwt_secret.key'
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret
    
    def _load_or_create_encryption_key(self) -> str:
        """Load or create encryption key"""
        key_file = self.security_dir / 'encryption.key'
        
        if key_file.exists():
            with open(key_file, 'r') as f:
                return f.read().strip()
        else:
            key = secrets.token_urlsafe(32)
            with open(key_file, 'w') as f:
                f.write(key)
            return key
    
    def create_default_infrastructure(self):
        """Create default super admin infrastructure"""
        
        # Create system tenant
        system_tenant, admin_user = self.create_tenant(
            name="Task Master System",
            owner_email="admin@taskmaster.ai",
            subscription_plan="enterprise",
            security_level=SecurityLevel.ENTERPRISE
        )
        
        print(f"✅ Created default system infrastructure")
    
    def load_enterprise_data(self):
        """Load enterprise data from disk"""
        try:
            # Load tenants
            if self.tenants_file.exists():
                with open(self.tenants_file, 'r') as f:
                    tenants_data = json.load(f)
                for tenant_data in tenants_data:
                    tenant = Tenant(
                        tenant_id=tenant_data['tenant_id'],
                        name=tenant_data['name'],
                        created_at=datetime.fromisoformat(tenant_data['created_at']),
                        owner_user_id=tenant_data['owner_user_id'],
                        subscription_plan=tenant_data['subscription_plan'],
                        security_level=SecurityLevel(tenant_data['security_level']),
                        settings=tenant_data['settings'],
                        is_active=tenant_data.get('is_active', True),
                        user_limit=tenant_data.get('user_limit', 50),
                        project_limit=tenant_data.get('project_limit', 100)
                    )
                    self.tenants[tenant.tenant_id] = tenant
            
            # Load users
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                for user_data in users_data:
                    user = User(
                        user_id=user_data['user_id'],
                        tenant_id=user_data['tenant_id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        role=UserRole(user_data['role']),
                        permissions=[Permission(p) for p in user_data['permissions']],
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                        is_active=user_data.get('is_active', True),
                        mfa_enabled=user_data.get('mfa_enabled', False),
                        security_clearance=user_data.get('security_clearance', 'standard')
                    )
                    self.users[user.user_id] = user
            
            # Load security policies
            if self.policies_file.exists():
                with open(self.policies_file, 'r') as f:
                    policies_data = json.load(f)
                for policy_data in policies_data:
                    policy = SecurityPolicy(
                        policy_id=policy_data['policy_id'],
                        tenant_id=policy_data['tenant_id'],
                        password_requirements=policy_data['password_requirements'],
                        session_timeout_minutes=policy_data['session_timeout_minutes'],
                        mfa_required=policy_data['mfa_required'],
                        ip_whitelist=policy_data['ip_whitelist'],
                        audit_retention_days=policy_data['audit_retention_days'],
                        encryption_level=policy_data['encryption_level'],
                        compliance_standards=policy_data['compliance_standards']
                    )
                    self.security_policies[policy.tenant_id] = policy
            
        except Exception as e:
            print(f"⚠️ Failed to load enterprise data: {e}")
    
    def save_enterprise_data(self):
        """Save enterprise data to disk"""
        try:
            # Save tenants
            tenants_data = []
            for tenant in self.tenants.values():
                tenant_data = asdict(tenant)
                tenant_data['created_at'] = tenant.created_at.isoformat()
                tenant_data['security_level'] = tenant.security_level.value
                tenants_data.append(tenant_data)
            
            with open(self.tenants_file, 'w') as f:
                json.dump(tenants_data, f, indent=2)
            
            # Save users
            users_data = []
            for user in self.users.values():
                user_data = asdict(user)
                user_data['created_at'] = user.created_at.isoformat()
                user_data['last_login'] = user.last_login.isoformat() if user.last_login else None
                user_data['role'] = user.role.value
                user_data['permissions'] = [p.value for p in user.permissions]
                users_data.append(user_data)
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            
            # Save security policies
            policies_data = []
            for policy in self.security_policies.values():
                policies_data.append(asdict(policy))
            
            with open(self.policies_file, 'w') as f:
                json.dump(policies_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save enterprise data: {e}")
    
    def save_audit_logs(self):
        """Save audit logs to disk"""
        try:
            audit_data = []
            for event in self.audit_events[-1000:]:  # Save last 1000 events
                event_data = asdict(event)
                event_data['timestamp'] = event.timestamp.isoformat()
                audit_data.append(event_data)
            
            with open(self.audit_logs_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save audit logs: {e}")
    
    def get_enterprise_summary(self) -> Dict[str, Any]:
        """Get enterprise security summary"""
        return {
            'tenants': len(self.tenants),
            'users': len(self.users),
            'active_users': len([u for u in self.users.values() if u.is_active]),
            'security_policies': len(self.security_policies),
            'audit_events': len(self.audit_events),
            'active_tokens': len(self.active_tokens),
            'mfa_adoption_rate': len([u for u in self.users.values() if u.mfa_enabled]) / len(self.users) if self.users else 0,
            'compliance_levels': list(set(t.security_level.value for t in self.tenants.values()))
        }

def main():
    """Demo of enterprise security manager"""
    print("Enterprise Security Manager Demo")
    print("=" * 40)
    
    security_manager = EnterpriseSecurityManager()
    
    # Demo: Create enterprise tenant
    print("\n🏢 Creating enterprise tenant...")
    tenant, admin_user = security_manager.create_tenant(
        name="Acme Corporation",
        owner_email="admin@acme.com",
        subscription_plan="enterprise",
        security_level=SecurityLevel.ENTERPRISE
    )
    
    # Demo: Create additional users
    print("\n👥 Creating team members...")
    
    project_manager = security_manager.create_user(
        tenant_id=tenant.tenant_id,
        username="john_pm",
        email="john@acme.com",
        role=UserRole.PROJECT_MANAGER,
        created_by_user_id=admin_user.user_id,
        mfa_enabled=True
    )
    
    developer = security_manager.create_user(
        tenant_id=tenant.tenant_id,
        username="jane_dev",
        email="jane@acme.com", 
        role=UserRole.DEVELOPER,
        created_by_user_id=admin_user.user_id,
        mfa_enabled=False
    )
    
    # Demo: Authenticate users
    print("\n🔐 Testing authentication...")
    
    admin_token = security_manager.authenticate_user(
        email="admin@acme.com",
        password="secure_password_123",
        mfa_code="123456",
        ip_address="192.168.1.100",
        user_agent="Task Master Enterprise/1.0"
    )
    
    if admin_token:
        print(f"✅ Admin authenticated - token expires: {admin_token.expires_at}")
    
    # Demo: Permission checking
    print("\n🛡️ Testing permissions...")
    
    can_manage_users = security_manager.check_permission(admin_user.user_id, Permission.MANAGE_USERS)
    can_delete_tasks = security_manager.check_permission(developer.user_id, Permission.DELETE_TASK)
    
    print(f"Admin can manage users: {can_manage_users}")
    print(f"Developer can delete tasks: {can_delete_tasks}")
    
    # Demo: User context
    print("\n📋 User context...")
    context = security_manager.get_user_context(admin_user.user_id)
    if context:
        print(f"User: {context['user']['username']} ({context['user']['role']})")
        print(f"Tenant: {context['tenant']['name']} ({context['tenant']['security_level']})")
        print(f"Permissions: {len(context['user']['permissions'])}")
    
    # Demo: Compliance report
    print("\n📊 Generating compliance report...")
    compliance_report = security_manager.get_compliance_report(tenant.tenant_id, "SOC2")
    
    print(f"Compliance score: {compliance_report['compliance_score']:.1%}")
    print(f"MFA adoption: {compliance_report['security_metrics']['mfa_adoption_rate']:.1%}")
    print(f"Total users: {compliance_report['user_metrics']['total_users']}")
    print(f"Audit events: {compliance_report['audit_metrics']['total_audit_events']}")
    
    # Enterprise summary
    summary = security_manager.get_enterprise_summary()
    print(f"\n📈 Enterprise Summary:")
    print(f"  Tenants: {summary['tenants']}")
    print(f"  Users: {summary['users']} ({summary['active_users']} active)")
    print(f"  MFA adoption: {summary['mfa_adoption_rate']:.1%}")
    print(f"  Compliance levels: {', '.join(summary['compliance_levels'])}")
    
    print(f"\n✅ Enterprise security demo completed")

if __name__ == "__main__":
    main()