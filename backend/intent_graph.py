"""
Graph-Based Intent Detection Layer for SP 21:2005 RAG System
"""
import re
from typing import Optional, Dict, Any

SECTION_KEYWORDS = {
    1: ["cement","concrete","aggregate","mortar","opc","portland","masonry block","precast",
        "pozzolana","slag cement","fence post","coping","kerb","cable cover","manhole",
        "ferrocement","roofing sheet","lintel","sill","autoclaved","aerated","brick",
        "burnt clay","fly ash","lime brick","sand lime","hollow block","paving block",
        "compressive strength","grade cement","33 grade","43 grade","53 grade","ready mixed",
        "fine aggregate","coarse aggregate","admixture","water cement","curing","rcc"],
    2: ["lime","building lime","hydraulic lime","hydrated lime","quicklime","calcium","putty"],
    3: ["stone","granite","marble","limestone","sandstone","slate","quartzite","laterite","flagstone","natural stone"],
    4: ["plywood","blockboard","particle board","fibre board","veneered","decorative plywood",
        "marine plywood","boiling water","mr grade","bwr grade","plywood type"],
    5: ["gypsum","plaster","plasterboard","gypsum board","gypsum partition"],
    6: ["timber","wood","sawn timber","structural timber","bamboo","log","pole","lumber",
        "species","teak","sal","deodar","seasoning"],
    7: ["bitumen","tar","asphalt","mastic","bituminous","cutback","emulsion","road tar","paving"],
    8: ["floor","wall covering","roof covering","tile","terrazzo","mosaic","ceramic tile",
        "flooring","roofing","finish","clay tile","mangalore"],
    9: ["waterproof","damp proof","water proof","waterproofing","damp proofing","moisture barrier"],
    10: ["sanitary","sanitary fitting","sanitary appliance","water fitting","faucet","cistern",
         "wash basin","urinal","water closet","toilet","sink","bath tub","shower","bidet",
         "valve","pipe fitting","bathroom","plumbing","lavatory","commode","flush valve"],
    11: ["hardware","builder hardware","hinge","latch","lock","handle","tower bolt","hasp","stay"],
    12: ["flush door","core board","batten door"],
    13: ["door","window","shutter","ventilator","door frame","window frame","glazing","chaukhat"],
    14: ["reinforcement","rebar","steel bar","deformed bar","wire fabric","welded mesh","tmt",
         "thermo mechanical","high strength deformed","mild steel bar","tor steel","binding wire"],
    15: ["structural steel","steel plate","steel sheet","mild steel","high tensile",
         "steel section","channel","angle","beam","joist","i-section","h-section"],
    16: ["aluminium","aluminum","light metal","alloy","aluminium alloy","duralumin","extrusion"],
    17: ["structural shape","i-beam","h-beam","channel section","angle section","t-section"],
    18: ["welding","electrode","welding electrode","welding wire","arc welding","gas welding",
         "flux","mig","tig","filler rod","coated electrode","submerged arc"],
    19: ["fastener","nut","screw","rivet","threaded","washer","stud"],
    20: ["wire rope","wire product","strand","barbed wire","chain link","fencing wire","galvanized wire"],
    21: ["glass","sheet glass","plate glass","wired glass","tempered glass","safety glass",
         "float glass","toughened","laminated glass","glass thickness"],
    22: ["filler","stopper","putty","sealing compound","caulking","mastic filler"],
    23: ["thermal insulation","insulation material","mineral wool","glass wool","rock wool",
         "expanded polystyrene","foam","insulating","heat insulation","u-value"],
}

CONTENT_TYPE_KEYWORDS = {
    "scope": ["scope","covers","what is","what are","about","purpose","application","applicable","what does","define"],
    "requirements": ["requirement","specification","physical requirement","chemical requirement",
                     "shall","minimum","maximum","not less than","not more than","limit","comply","standard specifies"],
    "physical": ["physical","compressive strength","tensile strength","flexural strength","density",
                 "water absorption","soundness","fineness","setting time","shrinkage","hardness",
                 "impact","abrasion","modulus","elongation","yield"],
    "chemical": ["chemical","composition","magnesia","sulphate","chloride","oxide","calcium","silica",
                 "alumina","iron oxide","loss on ignition","alkali","insoluble residue"],
    "dimensions": ["dimension","size","length","width","thickness","height","diameter","tolerance",
                   "nominal size","actual size","cross section"],
    "grading": ["grading","sieve","particle size","fineness modulus","zone","aggregate size","percentage passing"],
    "test": ["test method","testing","how to test","test for","method of test","specimen",
             "sampling","procedure","apparatus"],
    "classification": ["classification","class","grade","type","category","designation","kinds","types of","varieties"],
    "delivery": ["delivery","packing","packaging","bag","marking","labelling","storage"],
}

SECTION_NAMES = {
    1:"Cement and Concrete",2:"Building Limes",3:"Stones",4:"Wood Products for Building",
    5:"Gypsum Building Materials",6:"Timber",7:"Bitumen and Tar Products",
    8:"Floor Wall Roof Coverings and Finishes",9:"Water Proofing and Damp Proofing Materials",
    10:"Sanitary Appliances and Water Fittings",11:"Builders Hardware",12:"Wood Products",
    13:"Doors Windows and Shutters",14:"Concrete Reinforcement",15:"Structural Steels",
    16:"Light Metal and Their Alloys",17:"Structural Shapes",18:"Welding Electrodes and Wires",
    19:"Threaded Fasteners and Rivets",20:"Wire Ropes and Wire Products",21:"Glass",
    22:"Fillers Stoppers and Putties",23:"Thermal Insulation Materials",
}

def detect_is_code(query: str) -> Optional[str]:
    """Detect IS code references like 'IS 269', 'IS 269:1989', 'IS-8112'."""
    match = re.search(r'IS\s*[\:\-]?\s*(\d{2,5})', query, re.IGNORECASE)
    return match.group(1) if match else None

def detect_section(query: str) -> Optional[int]:
    query_lower = query.lower()
    scores = {}
    for sid, keywords in SECTION_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in query_lower:
                # Longer keyword matches get higher weight
                score += len(kw.split()) + (1 if len(kw) > 8 else 0)
        if score > 0:
            scores[sid] = score
    if not scores:
        return None
    best = max(scores, key=scores.get)
    # Only return None if all scores are very low and there's ambiguity
    if scores[best] <= 1 and len([s for s in scores if scores[s] == scores[best]]) > 3:
        return None
    return best

def detect_content_type(query: str) -> Optional[str]:
    query_lower = query.lower()
    scores = {}
    for ctype, keywords in CONTENT_TYPE_KEYWORDS.items():
        score = sum(len(kw.split()) for kw in keywords if kw in query_lower)
        if score > 0:
            scores[ctype] = score
    return max(scores, key=scores.get) if scores else None

def detect_intent(query: str) -> Dict[str, Any]:
    is_code = detect_is_code(query)
    section_id = detect_section(query)
    content_type = detect_content_type(query)

    if is_code:
        confidence, fallback = "high", "none"
    elif section_id and content_type:
        confidence, fallback = "high", "none"
    elif section_id or content_type:
        confidence, fallback = "medium", "section_only"
    else:
        confidence, fallback = "low", "full"

    return {
        "section_id": section_id,
        "section_name": SECTION_NAMES.get(section_id) if section_id else None,
        "content_type": content_type,
        "is_code_ref": is_code,
        "confidence": confidence,
        "fallback_mode": fallback,
    }
