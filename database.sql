-- Create the Database if you haven't already
CREATE DATABASE IF NOT EXISTS retail_sales_db;
USE retail_sales_db;

-- Create the Table matching the CSV structure
CREATE TABLE transactions (
    CustomerID INT,
    ProductID VARCHAR(50),
    Quantity INT,
    Price DECIMAL(10, 2),
    TransactionDate DATETIME,
    PaymentMethod VARCHAR(50),
    StoreLocation TEXT,
    ProductCategory VARCHAR(100),
    DiscountApplied_Percent DECIMAL(5, 2),
    TotalAmount DECIMAL(15, 2)
);