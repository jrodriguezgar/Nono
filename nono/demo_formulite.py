try:
    from genai_tasker.connector import connector_genai
except ImportError:
    # If running from nono directly and genai_tasker is not a package in path
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'genai_tasker'))
    from connector import connector_genai

# Ensure FormuLite is installed/loaded
if connector_genai.install_library("git+https://github.com/jrodriguezgar/FormuLite.git", "formulite"):
    import formulite
    from formulite import fxDate, fxString, fxExcel, fxNumeric
    
    print("\n--- FormuLite Demo ---")
    
    # 1. Date Operations
    print("\n[fxDate] Validation:")
    print(f"- Is 2025-02-30 valid? {fxDate.date_operations.is_valid_date('2025-02-30')}")
    print(f"- Is 2025-02-28 valid? {fxDate.date_operations.is_valid_date('2025-02-28')}")

    # 2. String Operations
    print("\n[fxString] Search:")
    text = "Programming is fun, programming is great"
    positions = fxString.string_operations.position_in_string(text, "is")
    print(f"- Positions of 'is' in '{text}': {positions}")
    
    # 3. Excel-Style Functions
    print("\n[fxExcel] VLOOKUP:")
    table = [
        ["ID", "Name", "Role"],
        [101, "Alice", "Developer"],
        [102, "Bob", "Designer"]
    ]
    role = fxExcel.VLOOKUP(101, table, 3)
    print(f"- VLOOKUP(101) -> {role}")

    # 4. Financial
    print("\n[fxNumeric] Future Value:")
    fv = fxNumeric.numeric_finance.future_value(rate=0.05, nper=10, pmt=-100, pv=-1000)
    print(f"- FV(0.05, 10, -100, -1000) = {fv:.2f}")

else:
    print("Failed to install FormuLite.")
