
import json


def company_names(authorships_json, limit=4):
    if isinstance(authorships_json, str):
        try:
            authorships = json.loads(authorships_json)
        except json.JSONDecodeError:
            return []
    elif isinstance(authorships_json, list):
        authorships = authorships_json
    else:
        return []

    if not isinstance(authorships, list):
        return []

    names = []
    seen = set()
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        for institution in authorship.get("institutions") or []:
            if not isinstance(institution, dict):
                continue
            if str(institution.get("type") or "").casefold() != "company":
                continue
            name = institution.get("display_name")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if name.casefold() not in seen:
                names.append(name)
                seen.add(name.casefold())
                if len(names) == limit:
                    return names
    return names
