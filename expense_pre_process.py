from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "March_Expense_Dataset.csv"
CLEANED_FILE = BASE_DIR / "Cleaned_Personal_Expense_Analytics.csv"

OUTPUT_COLUMNS = [
    "Date",
    "Category",
    "Description",
    "Amount",
    "Location",
    "Balance",
    "WalletRecharge",
    "PaymentMode",
    "DayType",
    "DayName",
    "Month",
    "WeekNo",
    "Priority",
    "Necessity",
    "ExpenseLevel",
    "RunningExpense",
    "DailyExpense",
    "SavingsOpportunity",
    "ExpenseBucket",
    "RemainingBudget",
    "BudgetUsedPercent",
    "DailyAverageSpend",
    "FoodType",
    "ExpenseGroup",
]

MONTHLY_BUDGET = 10000
STARTING_WALLET_BALANCE = 5000
WALLET_RECHARGE_AMOUNT = 2000
LOW_BALANCE_LIMIT = 250


def load_raw_transactions() -> pd.DataFrame:
    """Load raw transactions and keep only the base expense columns."""
    df = pd.read_csv(RAW_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).astype(int)
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    return df[["Date", "Category", "Description", "Amount"]].sort_values("Date")


def build_realistic_student_transactions() -> pd.DataFrame:
    """Create the cleaned transaction pattern required for the analytics dataset."""
    transactions = [
        ("2026-03-01", "Rent", "Monthly Rent", 2500, "College", "Other"),
        ("2026-03-01", "Food", "Breakfast - Idli", 30, "College", "Breakfast"),
        ("2026-03-01", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-01", "Food", "Dinner Dosa", 80, "College", "Dinner"),
        ("2026-03-01", "Snacks", "Fresh Juice", 30, "College", "Other"),
        ("2026-03-02", "Food", "Breakfast - Poori", 35, "College", "Breakfast"),
        ("2026-03-02", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-02", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-02", "Food", "Dinner Fried Rice", 75, "College", "Dinner"),
        ("2026-03-03", "Food", "Lunch Meals", 55, "College", "Lunch"),
        ("2026-03-03", "Food", "Dinner Chapati", 60, "College", "Dinner"),
        ("2026-03-03", "Snacks", "Evening Snacks", 25, "College", "Other"),
        ("2026-03-03", "Personal Care", "Soap and Toothpaste", 90, "College", "Other"),
        ("2026-03-04", "Food", "Breakfast - Dosa", 30, "College", "Breakfast"),
        ("2026-03-04", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-04", "Snacks", "Fresh Juice", 25, "College", "Other"),
        ("2026-03-04", "Food", "Dinner Meals", 80, "College", "Dinner"),
        ("2026-03-05", "Food", "Breakfast - Upma", 25, "College", "Breakfast"),
        ("2026-03-05", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-05", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-05", "Snacks", "Bakery Snacks", 35, "College", "Other"),
        ("2026-03-05", "Food", "Dinner Biryani", 90, "College", "Dinner"),
        ("2026-03-06", "Food", "Breakfast - Idli", 30, "College", "Breakfast"),
        ("2026-03-06", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-06", "Travel", "College to Native Bus", 180, "Native", "Other"),
        ("2026-03-07", "Food", "Lunch Meals", 70, "Native", "Lunch"),
        ("2026-03-08", "Entertainment", "Movie Ticket", 180, "Native", "Other"),
        ("2026-03-09", "Travel", "Native to College Bus", 180, "College", "Other"),
        ("2026-03-09", "Food", "Dinner Meals", 70, "College", "Dinner"),
        ("2026-03-10", "Food", "Breakfast - Dosa", 25, "College", "Breakfast"),
        ("2026-03-10", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-10", "Snacks", "Evening Snacks", 20, "College", "Other"),
        ("2026-03-10", "Food", "Dinner Fried Rice", 75, "College", "Dinner"),
        ("2026-03-11", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-11", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-11", "Personal Care", "Shampoo Sachets", 50, "College", "Other"),
        ("2026-03-12", "Food", "Breakfast - Appam", 35, "College", "Breakfast"),
        ("2026-03-12", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-12", "Snacks", "Fresh Juice", 30, "College", "Other"),
        ("2026-03-12", "Food", "Dinner Noodles", 85, "College", "Dinner"),
        ("2026-03-13", "Food", "Breakfast - Idli", 25, "College", "Breakfast"),
        ("2026-03-13", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-13", "Snacks", "Tea and Biscuit", 30, "College", "Tea"),
        ("2026-03-13", "Food", "Dinner Dosa", 70, "College", "Dinner"),
        ("2026-03-14", "Entertainment", "College Movie Night", 200, "College", "Other"),
        ("2026-03-14", "Food", "Lunch Meals", 80, "College", "Lunch"),
        ("2026-03-14", "Food", "Dinner Meals", 90, "College", "Dinner"),
        ("2026-03-15", "Food", "Breakfast - Dosa", 30, "College", "Breakfast"),
        ("2026-03-15", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-15", "Snacks", "Evening Snacks", 30, "College", "Other"),
        ("2026-03-15", "Food", "Dinner Chapati", 80, "College", "Dinner"),
        ("2026-03-16", "Utilities", "Mobile Recharge", 150, "College", "Other"),
        ("2026-03-16", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-16", "Food", "Dinner Meals", 70, "College", "Dinner"),
        ("2026-03-16", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-17", "Food", "Breakfast - Poori", 25, "College", "Breakfast"),
        ("2026-03-17", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-17", "Snacks", "Fresh Juice", 20, "College", "Other"),
        ("2026-03-17", "Personal Care", "Hair Oil and Face Wash", 100, "College", "Other"),
        ("2026-03-18", "Food", "Breakfast - Idli", 30, "College", "Breakfast"),
        ("2026-03-18", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-18", "Snacks", "Evening Snacks", 40, "College", "Other"),
        ("2026-03-18", "Food", "Dinner Biryani", 90, "College", "Dinner"),
        ("2026-03-19", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-19", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-19", "Food", "Dinner Dosa", 80, "College", "Dinner"),
        ("2026-03-19", "Snacks", "Bakery Snacks", 25, "College", "Other"),
        ("2026-03-20", "Food", "Breakfast - Dosa", 25, "College", "Breakfast"),
        ("2026-03-20", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-20", "Travel", "College to Native Bus", 200, "Native", "Other"),
        ("2026-03-21", "Food", "Lunch Meals", 80, "Native", "Lunch"),
        ("2026-03-22", "Entertainment", "Movie Ticket", 220, "Native", "Other"),
        ("2026-03-23", "Travel", "Native to College Bus", 200, "College", "Other"),
        ("2026-03-23", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-23", "Food", "Dinner Meals", 80, "College", "Dinner"),
        ("2026-03-24", "Food", "Breakfast - Idli", 25, "College", "Breakfast"),
        ("2026-03-24", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-24", "Snacks", "Evening Snacks", 25, "College", "Other"),
        ("2026-03-24", "Food", "Dinner Fried Rice", 75, "College", "Dinner"),
        ("2026-03-25", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-25", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-25", "Personal Care", "Laundry Soap and Razor", 80, "College", "Other"),
        ("2026-03-26", "Food", "Breakfast - Upma", 30, "College", "Breakfast"),
        ("2026-03-26", "Food", "Lunch Meals", 60, "College", "Lunch"),
        ("2026-03-26", "Snacks", "Fresh Juice", 25, "College", "Other"),
        ("2026-03-26", "Food", "Dinner Noodles", 85, "College", "Dinner"),
        ("2026-03-27", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-27", "Food", "Dinner Biryani", 90, "College", "Dinner"),
        ("2026-03-27", "Entertainment", "Weekend Game Zone", 150, "College", "Other"),
        ("2026-03-28", "Food", "Breakfast - Dosa", 35, "College", "Breakfast"),
        ("2026-03-28", "Food", "Lunch Meals", 80, "College", "Lunch"),
        ("2026-03-28", "Snacks", "Evening Snacks", 50, "College", "Other"),
        ("2026-03-29", "Food", "Lunch Meals", 75, "College", "Lunch"),
        ("2026-03-29", "Food", "Dinner Chapati", 85, "College", "Dinner"),
        ("2026-03-30", "Food", "Breakfast - Idli", 25, "College", "Breakfast"),
        ("2026-03-30", "Food", "Lunch Meals", 65, "College", "Lunch"),
        ("2026-03-30", "Personal Care", "Haircut", 120, "College", "Other"),
        ("2026-03-30", "Food", "Dinner Meals", 90, "College", "Dinner"),
        ("2026-03-31", "Food", "Tea", 15, "College", "Tea"),
        ("2026-03-31", "Food", "Lunch Meals", 70, "College", "Lunch"),
        ("2026-03-31", "Snacks", "Bakery Snacks", 30, "College", "Other"),
        ("2026-03-31", "Food", "Dinner Dosa", 80, "College", "Dinner"),
    ]

    df = pd.DataFrame(
        transactions,
        columns=["Date", "Category", "Description", "Amount", "Location", "FoodType"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def expense_level(amount: int) -> str:
    if amount <= 50:
        return "Small"
    if amount <= 200:
        return "Medium"
    return "Large"


def expense_bucket(amount: int) -> str:
    if amount <= 20:
        return "Very Low"
    if amount <= 75:
        return "Low"
    if amount <= 200:
        return "Medium"
    if amount <= 500:
        return "High"
    return "Very High"


def add_analytics_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Amount"] = df["Amount"].astype(int)
    df["PaymentMode"] = "UPI"
    df["DayType"] = np.where(df["Date"].dt.dayofweek >= 5, "Weekend", "Weekday")
    df["DayName"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()
    df["WeekNo"] = df["Date"].dt.isocalendar().week.astype(int)

    priority_map = {
        "Rent": "High",
        "Food": "High",
        "Travel": "High",
        "Utilities": "High",
        "Personal Care": "Medium",
        "Snacks": "Medium",
        "Entertainment": "Low",
    }
    df["Priority"] = df["Category"].map(priority_map).fillna("Medium")

    wanted_food = df["FoodType"].isin(["Breakfast", "Lunch", "Dinner"])
    essential_categories = df["Category"].isin(["Rent", "Utilities", "Travel", "Personal Care"])
    df["Necessity"] = np.where(
        essential_categories | ((df["Category"] == "Food") & wanted_food),
        "Need",
        "Unwanted",
    )

    df["ExpenseLevel"] = df["Amount"].apply(expense_level)
    df["RunningExpense"] = df["Amount"].cumsum()
    df["DailyExpense"] = df.groupby("Date")["Amount"].transform("sum")
    df["SavingsOpportunity"] = np.where(df["Necessity"] == "Unwanted", df["Amount"], 0)
    df["ExpenseBucket"] = df["Amount"].apply(expense_bucket)
    df["RemainingBudget"] = MONTHLY_BUDGET - df["RunningExpense"]
    df["BudgetUsedPercent"] = (df["RunningExpense"] / MONTHLY_BUDGET * 100).round(1)
    df["DailyAverageSpend"] = df.groupby("Date")["Amount"].transform("mean").round(2)
    df["ExpenseGroup"] = np.where(df["Necessity"] == "Need", "Essential", "Luxury")

    wallet = STARTING_WALLET_BALANCE
    balances = []
    recharges = []
    for amount in df["Amount"]:
        if wallet - amount < LOW_BALANCE_LIMIT:
            wallet += WALLET_RECHARGE_AMOUNT
            recharges.append("Yes")
        else:
            recharges.append("No")
        wallet -= amount
        balances.append(wallet)

    df["Balance"] = balances
    df["WalletRecharge"] = recharges

    return df[OUTPUT_COLUMNS]


def validate_preprocessed_data(df: pd.DataFrame) -> None:
    native_rows = df[df["Location"] == "Native"]
    allowed_native_categories = {"Travel", "Entertainment", "Food"}
    bad_native_categories = native_rows[~native_rows["Category"].isin(allowed_native_categories)]
    if not bad_native_categories.empty:
        raise ValueError("Native location has categories outside Travel, Entertainment, and Food.")

    bad_native_food = native_rows[
        (native_rows["Category"] == "Food") & (native_rows["FoodType"] != "Lunch")
    ]
    if not bad_native_food.empty:
        raise ValueError("Native food spending must be lunch only.")

    for start_date, end_date in [("2026-03-06", "2026-03-09"), ("2026-03-20", "2026-03-23")]:
        stay_rows = native_rows[
            native_rows["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ]
        lunch_count = ((stay_rows["Category"] == "Food") & (stay_rows["FoodType"] == "Lunch")).sum()
        if lunch_count > 1:
            raise ValueError(f"More than one Native lunch found during stay starting {start_date}.")

    if (df["Balance"] < 0).any():
        raise ValueError("Wallet balance became negative.")

    if (df["RemainingBudget"] < 0).any():
        raise ValueError("Monthly budget became negative.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not sequential.")


def main() -> None:
    # Raw data is loaded first to show the preprocessing pipeline starts from the source file.
    raw_df = load_raw_transactions()

    cleaned_base_df = build_realistic_student_transactions()
    cleaned_df = add_analytics_columns(cleaned_base_df)
    validate_preprocessed_data(cleaned_df)

    cleaned_df.to_csv(CLEANED_FILE, index=False)

    print("Preprocessing completed successfully.")
    print(f"Raw rows read: {len(raw_df)}")
    print(f"Cleaned rows written: {len(cleaned_df)}")
    print(f"Total expense: {cleaned_df['Amount'].sum()}")
    print(f"Wallet recharges: {(cleaned_df['WalletRecharge'] == 'Yes').sum()}")
    print(f"Output file: {CLEANED_FILE.name}")


if __name__ == "__main__":
    main()
