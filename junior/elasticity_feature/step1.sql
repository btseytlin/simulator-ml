-- SELECT * FROM transactions LIMIT 100

SELECT 
    sku,
    dates,
    price,
    count(user) as qty
FROM transactions
GROUP BY dates,
    sku,
    price
ORDER BY sku
