DROP TABLE IF EXISTS hw.bmi;

CREATE TABLE hw.bmi AS
WITH FirstVRawData AS (
	SELECT
		c.charttime,
		c.itemid,
		c.subject_id,
		c.icustay_id,
		CASE WHEN c.itemid IN (762, 763, 3723, 3580, 3581, 3582, 224639) THEN
			'WEIGHT'
		WHEN c.itemid IN (920, 1394, 4187, 3486, 3485, 4188, 226707, 226730) THEN
			'HEIGHT'
		END AS parameter,
		CASE WHEN c.itemid IN (3581) THEN
			c.valuenum * 0.45359237
		WHEN c.itemid IN (3582) THEN
			c.valuenum * 0.0283495231
		WHEN c.itemid IN (920, 1394, 4187, 3486, 226707) THEN
			c.valuenum * 2.54 / 100
		WHEN c.itemid IN (3485, 4188, 226730) THEN
			c.valuenum / 100
		ELSE
			c.valuenum
		END AS valuenum
	FROM
		mimiciii.chartevents c
	WHERE
		c.valuenum IS NOT NULL
		AND ((c.itemid IN (762, 763, 3723, 3580, 224639, -- Weight Kg
					3581, -- Weight lb
					3582, -- Weight oz
					920, 1394, 4187, 3486, 226707, -- Height inches
					3485, 4188, 226730 -- Height cm
)
				AND c.valuenum <> 0))
),
DemographicData AS (
	SELECT DISTINCT ON (a.subject_id)
		a.subject_id,
		p.gender,
		a.ethnicity,
		extract(YEAR FROM age(a.admittime, p.dob)) AS age
	FROM
		mimiciii.admissions a
		JOIN mimiciii.patients p ON a.subject_id = p.subject_id
),
SingleParameters AS (
	SELECT DISTINCT
		subject_id,
		icustay_id,
		parameter,
		first_value(valuenum) OVER (PARTITION BY subject_id,
			icustay_id,
			parameter ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_valuenum,
		min(valuenum) OVER (PARTITION BY subject_id,
			icustay_id,
			parameter ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS min_valuenum,
		max(valuenum) OVER (PARTITION BY subject_id,
			icustay_id,
			parameter ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_valuenum
	FROM
		FirstVRawData
),
PivotParameters AS (
	SELECT
		subject_id,
		icustay_id,
		max(
			CASE WHEN parameter = 'HEIGHT' THEN
				first_valuenum
			ELSE
				NULL
			END) AS height_first,
		max(
			CASE WHEN parameter = 'HEIGHT' THEN
				min_valuenum
			ELSE
				NULL
			END) AS height_min,
		max(
			CASE WHEN parameter = 'HEIGHT' THEN
				max_valuenum
			ELSE
				NULL
			END) AS height_max,
		max(
			CASE WHEN parameter = 'WEIGHT' THEN
				first_valuenum
			ELSE
				NULL
			END) AS weight_first,
		max(
			CASE WHEN parameter = 'WEIGHT' THEN
				min_valuenum
			ELSE
				NULL
			END) AS weight_min,
		max(
			CASE WHEN parameter = 'WEIGHT' THEN
				max_valuenum
			ELSE
				NULL
			END) AS weight_max
	FROM
		SingleParameters
	GROUP BY
		subject_id,
		icustay_id
)
SELECT
	f.icustay_id,
	f.subject_id,
	d.gender,
	d.ethnicity,
	d.age,
	round(cast(f.height_first AS numeric), 4) AS height_first,
	round(cast(f.height_min AS numeric), 4) AS height_min,
	round(cast(f.height_max AS numeric), 4) AS height_max,
	round(cast(f.weight_first AS numeric), 4) AS weight_first,
	round(cast(f.weight_min AS numeric), 4) AS weight_min,
	round(cast(f.weight_max AS numeric), 4) AS weight_max,
	round(cast(f.weight_first AS numeric) / (cast(f.height_first AS numeric) * cast(f.height_first AS numeric)), 4) AS bmi_first,
	round(cast(f.weight_min AS numeric) / (cast(f.height_max AS numeric) * cast(f.height_max AS numeric)), 4) AS bmi_min,
	round(cast(f.weight_max AS numeric) / (cast(f.height_min AS numeric) * cast(f.height_min AS numeric)), 4) AS bmi_max
FROM
	PivotParameters f
	JOIN DemographicData d ON d.subject_id = f.subject_id
ORDER BY
	subject_id,
	icustay_id;

SELECT
	*
FROM
	hw.bmi
LIMIT 100;

SELECT 
    gender,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height_first) AS median_height_first,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY weight_first) AS median_weight_first,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bmi_first) AS median_bmi_first,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height_min) AS median_height_min,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY weight_min) AS median_weight_min,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bmi_min) AS median_bmi_min,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height_max) AS median_height_max,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY weight_max) AS median_weight_max,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bmi_max) AS median_bmi_max
FROM hw.bmi
GROUP BY gender;

