-- Migration: Add category score/direction columns to extracted_facts, widen fact_type, drop category_scores.
-- Run with: psql "postgresql://user:pass@host:port/dbname" -f app/migrations/category_scores_to_columns.sql
-- Or paste into any PostgreSQL client (pgAdmin, DBeaver, etc.).

-- 1. Widen fact_type to VARCHAR(255) if it is shorter
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'fact_type'
      AND (character_maximum_length IS NULL OR character_maximum_length < 255)
  ) THEN
    ALTER TABLE extracted_facts
      ALTER COLUMN fact_type TYPE VARCHAR(255) USING fact_type::VARCHAR(255);
  END IF;
END $$;

-- 2. Add category columns (idempotent: only adds if missing)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'contacting_marketing_members_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN contacting_marketing_members_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'contacting_marketing_members_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN contacting_marketing_members_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'member_eligibility_molina_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN member_eligibility_molina_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'member_eligibility_molina_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN member_eligibility_molina_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'benefit_access_limitations_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN benefit_access_limitations_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'benefit_access_limitations_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN benefit_access_limitations_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'prior_authorization_required_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN prior_authorization_required_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'prior_authorization_required_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN prior_authorization_required_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claims_authorization_submissions_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN claims_authorization_submissions_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claims_authorization_submissions_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN claims_authorization_submissions_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'compliant_claim_requirements_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN compliant_claim_requirements_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'compliant_claim_requirements_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN compliant_claim_requirements_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claim_disputes_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN claim_disputes_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claim_disputes_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN claim_disputes_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'credentialing_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN credentialing_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'credentialing_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN credentialing_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claim_submission_important_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN claim_submission_important_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'claim_submission_important_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN claim_submission_important_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'coordination_of_benefits_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN coordination_of_benefits_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'coordination_of_benefits_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN coordination_of_benefits_direction REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'other_important_score') THEN
    ALTER TABLE extracted_facts ADD COLUMN other_important_score REAL NULL;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'other_important_direction') THEN
    ALTER TABLE extracted_facts ADD COLUMN other_important_direction REAL NULL;
  END IF;
END $$;

-- 3. Drop category_scores if it exists
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'extracted_facts' AND column_name = 'category_scores') THEN
    ALTER TABLE extracted_facts DROP COLUMN category_scores;
  END IF;
END $$;
