export const API_BASE_URL = 'http://localhost:8000';

export const DOCUMENT_TYPES = {
  lease_agreement: 'Lease Agreement',
  employment_contract: 'Employment Contract',
  service_agreement: 'Service Agreement',
  nda: 'Non-Disclosure Agreement',
  terms_of_service: 'Terms of Service',
  purchase_agreement: 'Purchase Agreement',
  loan_agreement: 'Loan Agreement',
  partnership_agreement: 'Partnership Agreement',
  vendor_contract: 'Vendor Contract',
  licensing_agreement: 'Licensing Agreement',
  insurance_policy: 'Insurance Policy',
  warranty: 'Warranty',
  other_legal: 'Other Legal Document',
  other: 'Other Document'
} as const;

export const RISK_LEVELS = {
  low: {
    label: 'Low Risk',
    color: 'text-success-green',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    threshold: 0.3
  },
  medium: {
    label: 'Medium Risk',
    color: 'text-warning-yellow',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    threshold: 0.6
  },
  high: {
    label: 'High Risk',
    color: 'text-danger-red',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    threshold: 0.8
  },
  urgent: {
    label: 'Urgent Review',
    color: 'text-white',
    bgColor: 'bg-danger-red',
    borderColor: 'border-danger-red',
    threshold: 1.0
  }
} as const;

export const LAWYER_URGENCY = {
  low: {
    label: 'Low Priority',
    description: 'Standard terms detected - basic review may be sufficient',
    color: 'text-success-green',
    icon: '✓'
  },
  medium: {
    label: 'Consider Review',
    description: 'Some concerning terms found - consider legal review',
    color: 'text-warning-yellow',
    icon: '⚠️'
  },
  high: {
    label: 'Recommended',
    description: 'Significant risks identified - legal consultation recommended',
    color: 'text-orange-600',
    icon: '⚡'
  },
  urgent: {
    label: 'Required',
    description: 'High-risk clauses detected - immediate legal review required',
    color: 'text-danger-red',
    icon: '🚨'
  }
} as const;

export const RISK_CATEGORIES = {
  financial_risk: {
    label: 'Financial Risk',
    description: 'Potential financial penalties, costs, or obligations',
    icon: '💰'
  },
  termination_risk: {
    label: 'Termination Risk',
    description: 'Difficulty in ending the agreement or unfavorable exit terms',
    icon: '🚪'
  },
  liability_risk: {
    label: 'Liability Risk',
    description: 'Personal or business liability exposure',
    icon: '⚖️'
  },
  renewal_risk: {
    label: 'Renewal Risk',
    description: 'Automatic renewals or lock-in mechanisms',
    icon: '🔄'
  },
  modification_risk: {
    label: 'Modification Risk',
    description: 'Ability to unilaterally change terms',
    icon: '📝'
  }
} as const;

export const CHAT_SUGGESTIONS = [
  "What are my main obligations in this document?",
  "Can I terminate this agreement early?",
  "What are the payment terms and deadlines?",
  "Are there any penalties or fees I should know about?",
  "What happens if I breach this contract?",
  "Are there any automatic renewal clauses?",
  "What are the most concerning parts of this document?",
  "Do I need a lawyer to review this?"
];

export const FILE_UPLOAD = {
  MAX_SIZE: 25 * 1024 * 1024, // 25MB
  ACCEPTED_TYPES: ['.pdf'],
  ACCEPTED_MIME_TYPES: ['application/pdf']
} as const;

export const ROUTES = {
  DASHBOARD: '/',
  DOCUMENTS: '/documents',
  DOCUMENT_VIEW: '/documents/:id',
  ANALYTICS: '/analytics'
} as const;