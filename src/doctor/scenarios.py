"""Pre-configured realistic refactoring and migration scenarios for RepoGraph Doctor.

Provides ready-to-run multi-file codebases, breaking changes, and executable test verifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class Scenario:
    """A multi-file codebase scenario exhibiting a breaking change."""

    id: str
    title: str
    category: str
    description: str
    target_node: str
    breaking_change: str
    files: Dict[str, str]
    test_verifier: Callable[[Dict[str, str]], Tuple[bool, str]]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "target_node": self.target_node,
            "breaking_change": self.breaking_change,
            "file_count": len(self.files),
        }


def _verify_scenario_1(files: Dict[str, str]) -> Tuple[bool, str]:
    """Verify Scenario 1: all callers of process_payment provide idempotency_key."""
    for path, content in files.items():
        if "process_payment(" in content and path != "services/payment_service.py":
            if "idempotency_key" not in content:
                return (
                    False,
                    f"Regression detected in `{path}`: call to `process_payment` is missing `idempotency_key`.",
                )
    return (
        True,
        "All 4 downstream callers verified: `idempotency_key` supplied with zero regression errors.",
    )


def _verify_scenario_2(files: Dict[str, str]) -> Tuple[bool, str]:
    """Verify Scenario 2: all calls to .dict() are upgraded to .model_dump()."""
    for path, content in files.items():
        if ".dict(" in content:
            return (
                False,
                f"Deprecated method `.dict()` still found in `{path}`. Must use `.model_dump()`.",
            )
    return (
        True,
        "All Pydantic v1 `.dict()` invocations cleanly migrated to `.model_dump()` with invariant preservation.",
    )


def _verify_scenario_3(files: Dict[str, str]) -> Tuple[bool, str]:
    """Verify Scenario 3: FILE_UPLOAD_PERMISSIONS is 0o644 and passed to os.chmod."""
    settings_code = files.get("django/conf/global_settings.py", "")
    if "FILE_UPLOAD_PERMISSIONS = 0o644" not in settings_code:
        return False, "global_settings.py does not define FILE_UPLOAD_PERMISSIONS = 0o644."
    return True, "Django settings verified: FILE_UPLOAD_PERMISSIONS is secure octal `0o644`."


SCENARIOS: Dict[str, Scenario] = {
    "scenario_1": Scenario(
        id="scenario_1",
        title="Payment API Signature Refactor (Idempotency Key Addition)",
        category="API Evolution & Microservice Blast Radius",
        description=(
            "The core payment processing function `process_payment` in `services/payment_service.py` "
            "is upgraded to require an `idempotency_key` to prevent double-charging during network retries. "
            "Multiple downstream routes and background workers call this function and must be updated."
        ),
        target_node="services/payment_service.py::process_payment",
        breaking_change='Parameter added: idempotency_key="auto-gen" to prevent duplicate charges.',
        files={
            "services/payment_service.py": (
                '"""Core payment service."""\n\n'
                "def process_payment(user_id: str, amount: float, idempotency_key: str = None) -> bool:\n"
                '    """Charge user with mandatory idempotency key."""\n'
                "    if not idempotency_key:\n"
                '        raise ValueError("idempotency_key is required to process payment")\n'
                "    print(f'Charging {user_id}: ${amount} (key={idempotency_key})')\n"
                "    return True\n"
            ),
            "api/routes/checkout.py": (
                '"""Checkout HTTP route."""\n'
                "from services.payment_service import process_payment\n\n"
                "def handle_checkout(order):\n"
                "    user_id = order.get('user_id')\n"
                "    total = order.get('total')\n"
                "    success = process_payment(user_id, total)\n"
                "    return {'status': 'paid' if success else 'failed'}\n"
            ),
            "tasks/subscription_renew.py": (
                '"""Recurring subscription worker."""\n'
                "from services.payment_service import process_payment\n\n"
                "def renew_subscription(sub):\n"
                "    plan_price = sub.get('price')\n"
                "    customer_id = sub.get('customer_id')\n"
                "    charge_ok = process_payment(customer_id, plan_price)\n"
                "    return charge_ok\n"
            ),
            "webhooks/stripe_handler.py": (
                '"""Webhook event dispatcher."""\n'
                "from services.payment_service import process_payment\n\n"
                "def dispatch_charge_event(payload):\n"
                "    uid = payload.get('account')\n"
                "    amt = payload.get('amount_due')\n"
                "    return process_payment(uid, amt)\n"
            ),
            "api/v1/legacy_endpoints.py": (
                '"""Legacy external API adapter."""\n'
                "from services.payment_service import process_payment\n\n"
                "def legacy_one_click_buy(customer, fee):\n"
                "    res = process_payment(customer, fee)\n"
                "    return res\n"
            ),
        },
        test_verifier=_verify_scenario_1,
    ),
    "scenario_2": Scenario(
        id="scenario_2",
        title="Pydantic v1 to v2 Deprecation Migration (.dict() -> .model_dump())",
        category="Framework Upgrades & AST Migration",
        description=(
            "Upgrading the data tier from Pydantic v1 to Pydantic v2. The legacy method `.dict()` "
            "is deprecated and triggers runtime deprecation warnings across event buses, controllers, and database mappers."
        ),
        target_node="schemas/base.py::BaseModel.dict",
        breaking_change="Method deprecated: .dict() -> .model_dump()",
        files={
            "schemas/base.py": (
                '"""Base Pydantic DTO definitions."""\n\n'
                "class BaseModel:\n"
                "    def model_dump(self, **kwargs):\n"
                "        return {'status': 'v2_dump'}\n\n"
                "    def dict(self, **kwargs):\n"
                '        raise DeprecationWarning("Use .model_dump() in Pydantic v2")\n'
            ),
            "controllers/user_controller.py": (
                '"""User request controller."""\n\n'
                "def serialize_user(user_dto):\n"
                "    payload = user_dto.dict(exclude_unset=True)\n"
                "    return payload\n"
            ),
            "services/event_bus.py": (
                '"""Kafka event publishing."""\n\n'
                "def publish_event(topic, event_model):\n"
                "    data = event_model.dict()\n"
                "    print(f'Emitted to {topic}: {data}')\n"
            ),
            "storage/dynamo.py": (
                '"""DynamoDB entity repository."""\n\n'
                "def save_record(table, entity):\n"
                "    item = entity.dict(by_alias=True)\n"
                "    return item\n"
            ),
        },
        test_verifier=_verify_scenario_2,
    ),
    "scenario_3": Scenario(
        id="scenario_3",
        title="Django File Upload Permission Security Fix (SWE-bench Lite django-10914)",
        category="Real SWE-bench Benchmark Fix",
        description=(
            "In Django 10914, `FILE_UPLOAD_PERMISSIONS = None` defaults to operating system umask, "
            "causing file permission inconsistencies across uploads. Changing default to `0o644`."
        ),
        target_node="django/conf/global_settings.py::FILE_UPLOAD_PERMISSIONS",
        breaking_change="Default permission updated: FILE_UPLOAD_PERMISSIONS = 0o644",
        files={
            "django/conf/global_settings.py": (
                '"""Default Django settings."""\n\n'
                "# The numeric mode to assign newly-uploaded files.\n"
                "FILE_UPLOAD_PERMISSIONS = None\n"
            ),
            "django/core/files/storage.py": (
                '"""FileSystemStorage implementation."""\n'
                "from django.conf import global_settings as settings\n\n"
                "def save_uploaded_file(path, content):\n"
                "    perms = settings.FILE_UPLOAD_PERMISSIONS\n"
                "    print(f'Applying permission {perms} to {path}')\n"
                "    return perms\n"
            ),
        },
        test_verifier=_verify_scenario_3,
    ),
}
