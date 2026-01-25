plutchik = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

# Emotion valence sets per index.tex Equation 6
# Surprise is excluded from valence sets and handled separately as informational novelty
POS_EMO = ["anticipation", "joy", "trust"]  # Positive valence emotions
NEG_EMO = ["anger", "disgust", "fear", "sadness"]  # Negative valence emotions
SURPRISE = "surprise"  # Handled separately as informational novelty indicator

topic_list = [
    'Conspiracy and Historical Events',
    'Disaster and Resilience',
    'Disputes and Opinions',
    'Economic and Financial Matters',
    'Education and Knowledge',
    'Environmental and Infrastructure',
    'Health and Social Welfare',
    'Labor and Workforce',
    'Leadership and Governance',
    'Legal and Regulatory Framework',
    'Miscellaneous and Other',
    'Power and Influence Dynamics',
    'Psychological and Emotional Aspects',
    'Quality of Life and Standards',
    'Security and Conflict',
    'Social and Cultural Aspects',
    'Sports and Recreation',
    'Trust, Credibility, and Communication',
]
path = r"/Users/briceshun/Documents/PhD 2022/youtube-emotion-foraging/data/parquet/"