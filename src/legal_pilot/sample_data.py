from __future__ import annotations

from legal_pilot.types import QAExample, SectionRecord


def build_sample_sections() -> list[SectionRecord]:
    return [
        SectionRecord(
            section_id="civ_1940",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1940",
            section_title="Hiring of real property",
            section_text=(
                "Hiring of real property is governed by this title. "
                "The obligations of landlords and tenants are defined by Section 1941 and Section 1941.1."
            ),
        ),
        SectionRecord(
            section_id="civ_1941",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1941",
            section_title="Landlord obligations",
            section_text=(
                "The lessor of a building intended for human occupation must put it into a condition fit for occupation, "
                "and repair all subsequent dilapidations, except as provided in Section 1941.2."
            ),
        ),
        SectionRecord(
            section_id="civ_1941_1",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1941.1",
            section_title="Tenantable building standards",
            section_text=(
                "A dwelling is tenantable only if it substantially complies with health and safety standards. "
                "Standards include effective waterproofing, plumbing, and heating."
            ),
        ),
        SectionRecord(
            section_id="civ_1941_2",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1941.2",
            section_title="Tenant-caused conditions",
            section_text=(
                "No repair duty arises when the dilapidation was caused by the tenant's lack of ordinary care, "
                "except to the extent otherwise required by Section 1942.4."
            ),
        ),
        SectionRecord(
            section_id="civ_1942_4",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1942.4",
            section_title="Demand for rent with uninhabitable conditions",
            section_text=(
                "A landlord may not demand rent if substantial habitability violations remain after notice, "
                "subject to the standards in Section 1941.1."
            ),
        ),
        SectionRecord(
            section_id="civ_1942_5",
            code_name="Civil Code",
            chapter="Landlord and Tenant",
            article="General Provisions",
            section_number="1942.5",
            section_title="Retaliatory conduct",
            section_text=(
                "A landlord may not retaliate against a tenant for lawful complaints about tenantability. "
                "Retaliation includes increasing rent or decreasing services after the tenant acts lawfully."
            ),
        ),
        SectionRecord(
            section_id="ccp_1161",
            code_name="Code of Civil Procedure",
            chapter="Unlawful Detainer",
            article="Grounds",
            section_number="1161",
            section_title="When tenant guilty of unlawful detainer",
            section_text=(
                "A tenant is guilty of unlawful detainer when the tenant continues in possession after default in rent "
                "or after failing to perform conditions of the lease."
            ),
        ),
        SectionRecord(
            section_id="ccp_1174_2",
            code_name="Code of Civil Procedure",
            chapter="Unlawful Detainer",
            article="Relief",
            section_number="1174.2",
            section_title="Relief against forfeiture",
            section_text=(
                "In an unlawful detainer action based on substandard conditions, the court may consider violations "
                "described in Section 1941.1 and Section 1942.4."
            ),
        ),
    ]


def build_sample_qa() -> list[QAExample]:
    return [
        QAExample(
            example_id="qa_train_1",
            split="train",
            question="Must a landlord keep a residential building fit for occupation?",
            choices=["Yes", "No"],
            answer_index=0,
            support_section_ids=["civ_1941"],
        ),
        QAExample(
            example_id="qa_train_2",
            split="train",
            question="Must a landlord repair damage caused by the tenant's lack of ordinary care?",
            choices=["Yes", "No"],
            answer_index=1,
            support_section_ids=["civ_1941_2"],
        ),
        QAExample(
            example_id="qa_test_1",
            split="test",
            question="Can a landlord raise rent to punish a tenant for making a lawful habitability complaint?",
            choices=["Yes", "No"],
            answer_index=1,
            support_section_ids=["civ_1942_5"],
        ),
        QAExample(
            example_id="qa_test_2",
            split="test",
            question="May a court in an unlawful detainer case consider tenantability violations from the Civil Code?",
            choices=["Yes", "No"],
            answer_index=0,
            support_section_ids=["ccp_1174_2", "civ_1941_1"],
        ),
    ]
