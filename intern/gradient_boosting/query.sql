SELECT 
    age,
    income,
    dependents,
    has_property,
    has_car,
    credit_score,
    job_tenure,
    has_education,
    loan_amount,
    dateDiff('day', loan_start, loan_deadline) as loan_period,
    greatest(dateDiff('day', loan_deadline, loan_payed),0) as delay_days
    -- loan_deadline - loan_start as loan_period,
    -- loan_payed - loan_deadline as delay_days
FROM default.loan_delay_days
