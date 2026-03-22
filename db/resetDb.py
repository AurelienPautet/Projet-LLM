from db import resetDbAndTables
import sys
import os

parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentDir)


if __name__ == "__main__":
    resetDbAndTables()
    print("Database reset completed.")
