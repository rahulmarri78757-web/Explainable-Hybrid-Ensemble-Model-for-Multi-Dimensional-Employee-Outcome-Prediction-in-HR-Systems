import sys
import os

# Add the 'backend' directory to sys.path so app modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.session import SessionLocal, engine, Base
from app.models.sql_models import User
from app.models.enums import UserRole
from app.core.security import get_password_hash

def init_db():
    # Make sure all tables are created
    Base.metadata.create_all(bind=engine)

def seed_users():
    db = SessionLocal()
    
    try:
        # Check if HR exists
        hr_user = db.query(User).filter(User.email == "hr@example.com").first()
        if not hr_user:
            hr_user = User(
                full_name="HR Admin",
                email="hr@example.com",
                hashed_password=get_password_hash("password"),
                role=UserRole.HR
            )
            db.add(hr_user)
            db.commit()
            db.refresh(hr_user)
            print("Created HR user")

        # Check if Manager exists
        manager_user = db.query(User).filter(User.email == "mike@example.com").first()
        if not manager_user:
            manager_user = User(
                full_name="Mike Manager",
                email="mike@example.com",
                hashed_password=get_password_hash("password"),
                role=UserRole.MANAGER
            )
            db.add(manager_user)
            db.commit()
            db.refresh(manager_user)
            print("Created Manager user")

        # Check if Employee exists
        employee_user = db.query(User).filter(User.email == "empm1@example.com").first()
        if not employee_user:
            employee_user = User(
                full_name="Employee One",
                email="empm1@example.com",
                hashed_password=get_password_hash("password"),
                role=UserRole.EMPLOYEE,
                manager_id=manager_user.id
            )
            db.add(employee_user)
            db.commit()
            print("Created Employee user")
            
        print("Database seeded successfully!")
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print("Seeding Initial Data...")
    seed_users()
