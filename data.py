# All static data for the SLA scheduling problem

TIMES = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]
TIME_IDX = {t: i for i, t in enumerate(TIMES)}

ROOMS = {
    "Beach 201":  18,
    "Beach 301":  25,
    "Frank 119":  95,
    "Loft 206":   55,
    "Loft 310":   48,
    "James 325":  110,
    "Roman 201":  40,
    "Roman 216":  80,
    "Slater 003": 32,
}
ROOM_NAMES = list(ROOMS.keys())
ROOM_CAPS  = [ROOMS[r] for r in ROOM_NAMES]

FACILITATORS = ["Lock", "Glen", "Banks", "Richards", "Shaw",
                "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

# Each activity: (name, enrollment, preferred, other)
ACTIVITIES = [
    ("SLA101A", 40, ["Glen","Lock","Banks"],           ["Numen","Richards","Shaw","Singer"]),
    ("SLA101B", 35, ["Glen","Lock","Banks"],           ["Numen","Richards","Shaw","Singer"]),
    ("SLA191A", 45, ["Glen","Lock","Banks"],           ["Numen","Richards","Shaw","Singer"]),
    ("SLA191B", 40, ["Glen","Lock","Banks"],           ["Numen","Richards","Shaw","Singer"]),
    ("SLA201",  60, ["Glen","Banks","Zeldin","Lock","Singer"], ["Richards","Uther","Shaw"]),
    ("SLA291",  50, ["Glen","Banks","Zeldin","Lock","Singer"], ["Richards","Uther","Shaw"]),
    ("SLA303",  25, ["Glen","Zeldin"],                 ["Banks"]),
    ("SLA304",  20, ["Singer","Uther"],                ["Richards"]),
    ("SLA394",  15, ["Tyler","Singer"],                ["Richards","Zeldin"]),
    ("SLA449",  30, ["Tyler","Zeldin","Uther"],        ["Zeldin","Shaw"]),
    ("SLA451",  90, ["Lock","Banks","Zeldin"],         ["Tyler","Singer","Shaw","Glen"]),
]

ACT_NAMES     = [a[0] for a in ACTIVITIES]
ACT_ENROLL    = [a[1] for a in ACTIVITIES]
ACT_PREFERRED = [a[2] for a in ACTIVITIES]
ACT_OTHER     = [a[3] for a in ACTIVITIES]

N_ACTIVITIES   = len(ACTIVITIES)
N_ROOMS        = len(ROOM_NAMES)
N_TIMES        = len(TIMES)
N_FACILITATORS = len(FACILITATORS)

FAC_IDX  = {f: i for i, f in enumerate(FACILITATORS)}
ROOM_IDX = {r: i for i, r in enumerate(ROOM_NAMES)}

# Pre-compute preferred/other indices for fast lookup
ACT_PREF_IDX  = [[FAC_IDX[f] for f in pref] for pref in ACT_PREFERRED]
ACT_OTHER_IDX = [[FAC_IDX[f] for f in oth]  for oth  in ACT_OTHER]

TYLER_IDX = FAC_IDX["Tyler"]

# Activity index helpers
SLA101 = [0, 1]   # SLA101A, SLA101B
SLA191 = [2, 3]   # SLA191A, SLA191B

BEACH_BUILDINGS  = {"Beach 201", "Beach 301"}
ROMAN_BUILDINGS  = {"Roman 201", "Roman 216"}
CLOSE_BUILDINGS  = BEACH_BUILDINGS | ROMAN_BUILDINGS
