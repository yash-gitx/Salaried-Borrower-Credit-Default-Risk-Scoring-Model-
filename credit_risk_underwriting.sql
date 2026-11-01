
-- QUERY 1: Underwriting Risk Matrix by CIBIL Band & FOIR Bucket
-- Highlights high-risk leakage inside "Approved" CIBIL tiers

WITH risk_cohorts AS (
    SELECT 
        loan_id,
        loan_amount_inr,
        default_90dpd,
        CASE 
            WHEN cibil_score < 680 THEN '1. Subprime (<680)'
            WHEN cibil_score BETWEEN 680 AND 719 THEN '2. Near-Prime (680-719)'
            WHEN cibil_score BETWEEN 720 AND 759 THEN '3. Prime (720-759)'
            ELSE '4. Super-Prime (760+)'
        END AS cibil_band,
        CASE 
            WHEN foir_pct <= 35.0 THEN 'Low FOIR (<=35%)'
            WHEN foir_pct <= 55.0 THEN 'Medium FOIR (35-55%)'
            ELSE 'High FOIR (>55%)'
        END AS foir_bucket
    FROM borrower_loans
)
SELECT 
    cibil_band,
    foir_bucket,
    COUNT(*) AS total_loans,
    ROUND(SUM(loan_amount_inr) / 10000000.0, 2) AS disbursed_exposure_cr,
    SUM(default_90dpd) AS bad_loans_90dpd,
    ROUND(100.0 * AVG(default_90dpd), 2) AS default_rate_pct
FROM risk_cohorts
GROUP BY cibil_band, foir_bucket
ORDER BY cibil_band, foir_bucket;

-- QUERY 2: Weight of Evidence (WoE) & Information Value (IV) for Inquiries
-- Standard credit risk feature selection methodology in SQL

WITH inquiry_bins AS (
    SELECT 
        CASE 
            WHEN inquiries_last_3m = 0 THEN '0 Inquiries'
            WHEN inquiries_last_3m = 1 THEN '1 Inquiry'
            WHEN inquiries_last_3m BETWEEN 2 AND 3 THEN '2-3 Inquiries'
            ELSE '4+ Inquiries (Credit Hungry)'
        END AS inquiry_bucket,
        default_90dpd
    FROM borrower_loans
),
bin_counts AS (
    SELECT 
        inquiry_bucket,
        COUNT(*) AS total_obs,
        SUM(CASE WHEN default_90dpd = 0 THEN 1 ELSE 0 END) AS good_count,
        SUM(CASE WHEN default_90dpd = 1 THEN 1 ELSE 0 END) AS bad_count
    FROM inquiry_bins
    GROUP BY inquiry_bucket
),
distributions AS (
    SELECT 
        inquiry_bucket,
        total_obs,
        good_count,
        bad_count,
        1.0 * good_count / SUM(good_count) OVER () AS dist_good,
        1.0 * bad_count / SUM(bad_count) OVER () AS dist_bad
    FROM bin_counts
)
SELECT 
    inquiry_bucket,
    total_obs,
    ROUND(100.0 * bad_count / total_obs, 2) AS bad_rate_pct,
    ROUND(LN(dist_good / NULLIF(dist_bad, 0)), 4) AS woe,
    ROUND((dist_good - dist_bad) * LN(dist_good / NULLIF(dist_bad, 0)), 4) AS iv_contribution
FROM distributions
ORDER BY bad_rate_pct ASC;


-- QUERY 3: Hidden Prime Opportunity (Near-Prime CIBIL 675-699 with Strong Cashflow)

SELECT 
    employer_category,
    COUNT(*) AS rejected_by_legacy_cutoff,
    ROUND(AVG(cibil_score), 0) AS avg_cibil,
    ROUND(AVG(foir_pct), 1) AS avg_foir_pct,
    ROUND(100.0 * AVG(default_90dpd), 2) AS actual_90dpd_rate_pct,
    ROUND(SUM(loan_amount_inr) / 10000000.0, 2) AS safe_disbursal_opportunity_cr
FROM borrower_loans
WHERE cibil_score BETWEEN 675 AND 699
  AND foir_pct <= 35.0
  AND nach_bounce_12m = 0
  AND inquiries_last_3m <= 1
GROUP BY employer_category
ORDER BY actual_90dpd_rate_pct ASC;