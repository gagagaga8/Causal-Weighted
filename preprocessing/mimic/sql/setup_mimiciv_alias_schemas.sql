-- ========================================
-- Create mimiciv_* alias schemas and views in local MIMIC database
-- Purpose: allow official MIMIC-IV concept SQL to run on local mimic_* schemas.
-- Migrated from temp code to preprocessing dir as prerequisite setup.
-- ========================================

-- 1. Createtarget schema ifnot in 
CREATE SCHEMA IF NOT EXISTS mimiciv_hosp;
CREATE SCHEMA IF NOT EXISTS mimiciv_icu;
CREATE SCHEMA IF NOT EXISTS mimiciv_ed;
CREATE SCHEMA IF NOT EXISTS mimiciv_derived;

-- 2. as mimic_hosp inAll TableCreate to mimiciv_hosp
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'mimic_hosp'
          AND table_type = 'BASE TABLE'
    ) LOOP
        EXECUTE format(
            'CREATE OR REPLACE VIEW mimiciv_hosp.%I AS SELECT * FROM mimic_hosp.%I;',
            r.table_name, r.table_name
        );
    END LOOP;
END $$;

-- 3. as mimic_icu inAll TableCreate to mimiciv_icu
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'mimic_icu'
          AND table_type = 'BASE TABLE'
    ) LOOP
        EXECUTE format(
            'CREATE OR REPLACE VIEW mimiciv_icu.%I AS SELECT * FROM mimic_icu.%I;',
            r.table_name, r.table_name
        );
    END LOOP;
END $$;

-- 4. as mimic_ed if in Create to mimiciv_ed
DO $$
DECLARE
    r RECORD;
    ed_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimic_ed'
    ) INTO ed_exists;

    IF ed_exists THEN
        FOR r IN (
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'mimic_ed'
              AND table_type = 'BASE TABLE'
        ) LOOP
            EXECUTE format(
                'CREATE OR REPLACE VIEW mimiciv_ed.%I AS SELECT * FROM mimic_ed.%I;',
                r.table_name, r.table_name
            );
        END LOOP;
    END IF;
END $$;

-- mimiciv_derived asOutput schema Create No need 
