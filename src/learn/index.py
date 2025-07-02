import json
import os
from datetime import datetime, date

# File to store medicine records
DATA_FILE = 'medicines.json'

def load_data():
    """Load medicine data from JSON file."""
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data):
    """Save medicine data to JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def add_medicine(data):
    """Add a new medicine record."""
    name = input("Enter medicine name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    try:
        quantity = int(input("Enter quantity: "))
    except ValueError:
        print("Quantity must be a number.")
        return
    exp_input = input("Enter expiration date (YYYY-MM-DD): ")
    try:
        exp_date = datetime.strptime(exp_input, '%Y-%m-%d').date()
    except ValueError:
        print("Date format should be YYYY-MM-DD.")
        return

    record = {
        "name": name,
        "quantity": quantity,
        "expiration": exp_date.isoformat()
    }
    data.append(record)
    save_data(data)
    print(f"Added {name} successfully.")


def view_medicines(data):
    """Display all medicine records."""
    if not data:
        print("No medicines found.")
        return
    print(f"\n{'ID':<3} {'Name':<20} {'Qty':<5} {'Expiration':<10}")
    print("-" * 40)
    for idx, med in enumerate(data, 1):
        print(f"{idx:<3} {med['name']:<20} {med['quantity']:<5} {med['expiration']}" )
    print()


def search_medicine(data):
    """Search for medicine by name keyword."""
    keyword = input("Enter name to search: ").strip().lower()
    results = [med for med in data if keyword in med['name'].lower()]
    if not results:
        print("No matching medicines found.")
        return
    print(f"\n{'ID':<3} {'Name':<20} {'Qty':<5} {'Expiration':<10}")
    print("-" * 40)
    for idx, med in enumerate(results, 1):
        print(f"{idx:<3} {med['name']:<20} {med['quantity']:<5} {med['expiration']}")
    print()


def update_medicine(data):
    """Update an existing medicine record."""
    view_medicines(data)
    try:
        idx = int(input("Enter ID of medicine to update: ")) - 1
        if idx < 0 or idx >= len(data):
            print("Invalid ID.")
            return
    except ValueError:
        print("Invalid input.")
        return

    med = data[idx]
    print(f"Updating {med['name']} (leave blank to keep current)")
    new_name = input(f"New name [{med['name']}]: ").strip() or med['name']
    qty_input = input(f"New quantity [{med['quantity']}]: ").strip()
    if qty_input:
        try:
            new_qty = int(qty_input)
        except ValueError:
            print("Quantity must be a number. Update cancelled.")
            return
    else:
        new_qty = med['quantity']

    exp_input = input(f"New expiration [{med['expiration']}]: ").strip()
    if exp_input:
        try:
            new_exp = datetime.strptime(exp_input, '%Y-%m-%d').date().isoformat()
        except ValueError:
            print("Date format should be YYYY-MM-DD. Update cancelled.")
            return
    else:
        new_exp = med['expiration']

    data[idx] = {"name": new_name, "quantity": new_qty, "expiration": new_exp}
    save_data(data)
    print("Medicine updated successfully.")


def delete_medicine(data):
    """Delete a medicine record."""
    view_medicines(data)
    try:
        idx = int(input("Enter ID of medicine to delete: ")) - 1
        if idx < 0 or idx >= len(data):
            print("Invalid ID.")
            return
    except ValueError:
        print("Invalid input.")
        return

    med = data.pop(idx)
    save_data(data)
    print(f"Deleted {med['name']} successfully.")


def check_expired(data):
    """Check and list expired medicines."""
    today = date.today()
    expired = [med for med in data if datetime.strptime(med['expiration'], '%Y-%m-%d').date() < today]
    if not expired:
        print("No expired medicines.")
        return
    print("Expired medicines:")
    print(f"{'Name':<20} {'Qty':<5} {'Expiration':<10}")
    print("-" * 40)
    for med in expired:
        print(f"{med['name']:<20} {med['quantity']:<5} {med['expiration']}")
    print()


def main():
    """Main menu loop for the console-based system."""
    data = load_data()

    while True:
        print("""
====== Medicine Inventory System ======
1. Add Medicine
2. View Medicines
3. Search Medicine
4. Update Medicine
5. Delete Medicine
6. Check Expired Medicines
7. Exit
""")
        choice = input("Enter choice [1-7]: ")
        if choice == '1':
            add_medicine(data)
        elif choice == '2':
            view_medicines(data)
        elif choice == '3':
            search_medicine(data)
        elif choice == '4':
            update_medicine(data)
        elif choice == '5':
            delete_medicine(data)
        elif choice == '6':
            check_expired(data)
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == '__main__':
    main()
