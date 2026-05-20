# %% [markdown]
# # 02 — Pydantic v2 Patterns
#
# Pydantic v2 (rewritten in Rust) is ~5–50× faster than v1.
# Key new concepts: `model_validator`, `field_validator`, discriminated unions, `TypeAdapter`.

# %%
from pydantic import BaseModel, Field, field_validator, model_validator, TypeAdapter
from typing import Annotated, Literal
from datetime import datetime, timezone, timedelta

# ── 1. field_validator ────────────────────────────────────────────────────────
# Runs on a single field after its type is coerced.
# Use @classmethod — the first arg is the class, not an instance.

class UserCreate(BaseModel):
    username: str
    email: str
    password: str = Field(min_length=8)

    @field_validator("username")
    @classmethod
    def lowercase_username(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("must contain @")
        return v.lower()

user = UserCreate(username="  Alice  ", email="Alice@Example.COM", password="secret123")
print(user)  # username='alice', email='alice@example.com'

# %%
# ── 2. model_validator — cross-field validation ───────────────────────────────
# model_validator(mode="after") runs after all field validators succeed.
# `self` is already the validated model instance.

class PasswordChange(BaseModel):
    new_password: str = Field(min_length=8)
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self) -> "PasswordChange":
        if self.new_password != self.confirm_password:
            raise ValueError("passwords do not match")
        return self

try:
    PasswordChange(new_password="securepass", confirm_password="different")
except Exception as e:
    print(f"validation error: {e}")

PasswordChange(new_password="securepass", confirm_password="securepass")
print("match ✓")

# %%
# ── 3. Discriminated unions ───────────────────────────────────────────────────
# Instead of Union[A, B, C] (tries each in order — slow, ambiguous),
# a discriminated union uses a literal "type" field to select the model instantly.

class EmailNotification(BaseModel):
    kind: Literal["email"]
    to: str
    subject: str
    body: str

class PushNotification(BaseModel):
    kind: Literal["push"]
    device_token: str
    title: str

class SMSNotification(BaseModel):
    kind: Literal["sms"]
    phone: str
    message: str

Notification = Annotated[
    EmailNotification | PushNotification | SMSNotification,
    Field(discriminator="kind"),
]

ta = TypeAdapter(Notification)

n1 = ta.validate_python({"kind": "email", "to": "a@b.com", "subject": "Hi", "body": "Hello"})
n2 = ta.validate_python({"kind": "push", "device_token": "abc", "title": "Alert"})

print(type(n1).__name__, type(n2).__name__)

# %%
# ── 4. TypeAdapter — validate outside a model ─────────────────────────────────
# Validate arbitrary Python types without wrapping in a BaseModel.

from pydantic import ValidationError

PositiveInt = TypeAdapter(Annotated[int, Field(gt=0)])
TagList = TypeAdapter(list[str])

print(PositiveInt.validate_python(42))    # 42
try:
    PositiveInt.validate_python(-1)
except ValidationError as e:
    print("rejected:", e.error_count(), "error(s)")

# Great for validating config files, API responses from external services, etc.
raw_tags = ["Python", "FastAPI", ""]
clean = TagList.validate_python([t for t in raw_tags if t])
print(clean)

# %%
# ── 5. from_attributes (ORM mode) ────────────────────────────────────────────
# Allows constructing a Pydantic model directly from an ORM object.
# Pydantic reads attributes instead of dict keys.

from pydantic import ConfigDict

class ItemResponse(BaseModel):
    id: int
    name: str
    model_config = ConfigDict(from_attributes=True)

class FakeOrmItem:  # simulates a SQLAlchemy model
    id = 1
    name = "Widget"

item = ItemResponse.model_validate(FakeOrmItem())
print(item)

# %%
# ── Exercise ──────────────────────────────────────────────────────────────────
# Define a `PaymentRequest` model that:
#   1. Has `amount_cents: int` (must be > 0)
#   2. Has `currency: str` (must be one of "usd", "eur", "gbp") using Literal
#   3. Has `idempotency_key: str` (UUID format — use a field_validator)
#   4. Uses model_validator to reject amounts > 1_000_000 cents for "gbp"
