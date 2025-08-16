# backend/app/services/risk_assessment.py

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class AdvancedRiskAssessment:
    def __init__(self):
        self.enterprise_risk_patterns = {
            'financial_risk': {
                'patterns': [
                    # Enterprise financial risks
                    r'late fee.*\d+(?:\.\d+)?%', r'interest.*\d+(?:\.\d+)?%', r'penalty.*\$\d+',
                    r'liquidated damages', r'liability cap.*\$\d+', r'maximum liability.*\$\d+',
                    r'non-refundable', r'forfeiture', r'overage charge', r'price increase',
                    r'auto.?renew', r'automatic.*renew', r'subscription fee increase',
                    r'additional charges', r'supplemental fees', r'professional services fee'
                ],
                'weight': 0.25
            },
            'termination_risk': {
                'patterns': [
                    # Enterprise termination risks
                    r'immediate termination', r'terminate.*without.*cause', r'terminate.*immediately',
                    r'terminate.*without.*notice', r'suspend.*access', r'suspend.*service',
                    r'right to suspend', r'at.*sole.*discretion', r'termination for convenience',
                    r'material breach', r'cure period', r'30.*days.*cure', r'opportunity to cure'
                ],
                'weight': 0.20
            },
            'liability_risk': {
                'patterns': [
                    # Enterprise liability risks
                    r'customer.*indemnify', r'customer.*indemnification', r'customer.*shall.*indemnify',
                    r'defend.*indemnify.*hold.*harmless', r'indemnify.*against.*all', r'broad.*indemnification',
                    r'unlimited.*liability', r'consequential.*damages', r'indirect.*damages',
                    r'punitive.*damages', r'special.*damages', r'liability.*exclusion',
                    r'disclaim.*liability', r'no.*liability.*for', r'exclude.*liability'
                ],
                'weight': 0.30  # Higher weight for enterprise liability
            },
            'data_privacy_risk': {
                'patterns': [
                    # Data and privacy risks
                    r'data.*breach', r'security.*breach', r'customer.*data', r'personal.*data',
                    r'gdpr', r'hipaa', r'privacy.*laws', r'data.*processing', r'data.*retention',
                    r'delete.*data', r'export.*data', r'data.*portability', r'right.*to.*be.*forgotten'
                ],
                'weight': 0.15
            },
            'compliance_risk': {
                'patterns': [
                    # Compliance and regulatory risks
                    r'regulatory.*compliance', r'audit.*rights', r'compliance.*obligations',
                    r'export.*control', r'trade.*restrictions', r'sanctions', r'embargo',
                    r'criminal.*liability', r'regulatory.*violations', r'legal.*compliance'
                ],
                'weight': 0.10
            }
        }

    def calculate_comprehensive_risk(self, text: str) -> tuple[float, Dict[str, float], List[str]]:
        """Enhanced risk assessment for enterprise documents"""
        text_lower = text.lower()
        risk_breakdown = {}
        identified_risks = []
        total_risk = 0.0

        # Document length factor (enterprise docs often have higher base risk)
        word_count = len(text.split())
        document_sophistication = min(word_count / 5000, 1.0)  # Normalize at 5000 words

        # Enterprise document indicators
        enterprise_indicators = [
            'saas', 'software as a service', 'enterprise', 'subscription',
            'indemnification', 'limitation of liability', 'governing law'
        ]
        enterprise_count = sum(1 for indicator in enterprise_indicators if indicator in text_lower)
        enterprise_factor = min(enterprise_count / 10, 0.3)  # Up to 30% boost for enterprise features

        for risk_category, config in self.enterprise_risk_patterns.items():
            category_risk = 0.0
            category_matches = []

            for pattern in config['patterns']:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    category_matches.extend(matches)
                    # Progressive risk increase per match (diminishing returns)
                    match_count = len(matches)
                    if match_count == 1:
                        category_risk += 0.3
                    elif match_count == 2:
                        category_risk += 0.5
                    elif match_count >= 3:
                        category_risk += 0.7
                    logger.info(f"Risk pattern matched in {risk_category}: {pattern} -> {matches}")

            # Apply enterprise and sophistication factors
            if category_risk > 0:
                category_risk += enterprise_factor * 0.5  # Boost risk for enterprise docs
                category_risk += document_sophistication * 0.2  # Boost for sophisticated docs

            # Cap category risk at 1.0
            category_risk = min(category_risk, 1.0)
            risk_breakdown[risk_category] = category_risk

            # Add to total weighted risk
            total_risk += category_risk * config['weight']

            # Track specific risks found
            if category_risk > 0:
                risk_level = "high" if category_risk > 0.7 else "medium" if category_risk > 0.4 else "low"
                identified_risks.append(f"{risk_category.replace('_', ' ').title()} ({risk_level})")

        # Apply base enterprise risk floor
        if enterprise_count > 3:  # Clear enterprise document
            total_risk = max(total_risk, 0.25)  # Minimum 25% risk for enterprise docs

        # Special handling for indemnification clauses
        if any('indemnif' in text_lower for pattern in ['indemnif', 'hold harmless']):
            total_risk += 0.15  # Significant boost for indemnification

        # Special handling for liability caps
        if any(pattern in text_lower for pattern in ['liability cap', 'limitation of liability', 'maximum liability']):
            # Liability caps can be good (reduce risk) or bad (if they're very high)
            liability_cap_matches = re.findall(r'liability.*\$[\d,]+', text_lower)
            if liability_cap_matches:
                total_risk += 0.1  # Moderate increase for specific liability amounts
            else:
                total_risk += 0.05  # Small increase for general liability limitations

        # Ensure minimum risk for complex documents
        if not identified_risks:
            total_risk = max(total_risk, 0.1)  # Always at least 10% risk
        else:
            total_risk = min(total_risk, 0.95)  # Cap at 95%

        logger.info(f"Enhanced risk calculation: Total={total_risk:.3f}, Enterprise factor={enterprise_factor:.3f}, "
                   f"Sophistication={document_sophistication:.3f}, Risks={identified_risks}")

        return total_risk, risk_breakdown, identified_risks

class LawyerRecommendationEngine:
    def __init__(self):
        self.enterprise_thresholds = {
            'urgent': 0.7,      # Enterprise docs with high risk need urgent review
            'high': 0.4,        # Lower threshold for enterprise docs
            'medium': 0.25,     # Even moderate enterprise complexity needs review
            'low': 0.15         # Very simple enterprise docs
        }

    def generate_recommendation(self, risk_score: float, complexity_score: float,
                          document_type: str, risk_breakdown: Dict[str, float],
                          urgency_signals: List[str] = None) -> tuple[bool, str, str]:
        """
        Enhanced lawyer recommendation for enterprise documents
        Returns: (should_recommend, urgency_level, reasoning)
        """
        logger.info(f"Generating enterprise recommendation: risk={risk_score:.3f}, complexity={complexity_score:.3f}, "
                   f"type={document_type}, urgency_signals={len(urgency_signals or [])}")

        # Calculate composite score with enterprise bias
        composite_score = (risk_score * 0.7) + (complexity_score * 0.3)  # Weight risk more heavily

        # Enterprise document type modifiers
        enterprise_types = ['service_agreement', 'saas_agreement', 'subscription_agreement', 'enterprise_agreement']
        high_risk_types = ['employment_contract', 'service_agreement', 'purchase_agreement', 'saas_agreement']

        if document_type in enterprise_types:
            composite_score += 0.15  # Significant boost for enterprise agreements
        elif document_type in high_risk_types:
            composite_score += 0.1

        # Enhanced risk category analysis
        critical_enterprise_risks = ['liability_risk', 'data_privacy_risk', 'termination_risk']
        critical_risk_score = sum(risk_breakdown.get(risk, 0) for risk in critical_enterprise_risks) / len(critical_enterprise_risks)

        if critical_risk_score > 0.5:
            composite_score += 0.2  # Major boost for critical enterprise risks

        # Indemnification detection (high priority for enterprise)
        if urgency_signals:
            indemnification_signals = [signal for signal in urgency_signals if 'indemnif' in signal.lower()]
            liability_signals = [signal for signal in urgency_signals if 'liability' in signal.lower()]

            if indemnification_signals:
                composite_score += 0.25  # Major boost for indemnification
            if liability_signals:
                composite_score += 0.15  # Boost for liability issues

            # General urgency signal boost
            urgency_boost = min(len(urgency_signals) * 0.05, 0.2)
            composite_score += urgency_boost

        # Enterprise-specific criteria
        enterprise_complexity_threshold = 0.4  # Lower threshold for enterprise docs
        if complexity_score > enterprise_complexity_threshold:
            composite_score += 0.1

        # Determine recommendation with enterprise-adjusted thresholds
        if composite_score >= self.enterprise_thresholds['urgent']:
            recommendation = (True, 'urgent',
                            'Enterprise agreement with high-risk provisions including indemnification, liability, or termination clauses - immediate legal review required')
        elif composite_score >= self.enterprise_thresholds['high']:
            recommendation = (True, 'high',
                            'Significant enterprise risks identified including complex liability, data, or compliance provisions - legal consultation strongly recommended')
        elif composite_score >= self.enterprise_thresholds['medium']:
            recommendation = (True, 'medium',
                            'Enterprise agreement with moderate complexity - legal review recommended to understand obligations and risks')
        else:
            # Even "low" risk enterprise docs should usually get review
            if complexity_score > 0.3 or risk_score > 0.2:
                recommendation = (True, 'low',
                                'Enterprise agreement detected - basic legal review recommended for peace of mind')
            else:
                recommendation = (False, 'low',
                                'Simple agreement with standard terms - basic review may be sufficient')

        logger.info(f"Enterprise recommendation: {recommendation}")
        return recommendation

def calculate_overall_risk_from_breakdown(risk_breakdown: dict) -> float:
    """Calculate overall risk score from risk breakdown with enterprise weighting"""
    enterprise_weights = {
        'financial_risk': 0.25,
        'termination_risk': 0.20,
        'liability_risk': 0.30,  # Higher weight for liability in enterprise
        'data_privacy_risk': 0.15,  # New category for enterprise
        'compliance_risk': 0.10,   # New category for enterprise
        # Fallback weights for older risk categories
        'renewal_risk': 0.10,
        'modification_risk': 0.10
    }

    total_risk = 0.0
    total_weight = 0.0

    for risk_type, risk_value in risk_breakdown.items():
        weight = enterprise_weights.get(risk_type, 0.05)  # Default small weight for unknown categories
        total_risk += risk_value * weight
        total_weight += weight

    # Normalize if we don't have full weight coverage
    if total_weight < 1.0 and total_weight > 0:
        total_risk = total_risk / total_weight

    return min(total_risk, 1.0)