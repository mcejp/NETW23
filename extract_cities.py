import json

def load_geonames(path):
    """Yield GeoNames records from a tab-separated file.
    Assumes standard geonames columns where latitude is column 5 and longitude is column 6 (0-based indices 4,5)
    and population is at column 15 (1-based) -> index 14.
    """
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 6:
                continue
            try:
                lat = float(cols[4])
                lon = float(cols[5])
            except Exception:
                continue
            # parse population if available
            pop = 0
            if len(cols) > 14:
                try:
                    pop = int(cols[14])
                except Exception:
                    pop = 0
            # parse feature class/code if available
            feature_class = cols[6] if len(cols) > 6 else ''
            feature_code = cols[7] if len(cols) > 7 else ''
            yield {
                'geonameid': cols[0] if len(cols) > 0 else '',
                'name': cols[1] if len(cols) > 1 else '',
                'asciiname': cols[2] if len(cols) > 2 else '',
                'alternatenames': cols[3] if len(cols) > 3 else '',
                'lat': lat,
                'lon': lon,
                'population': pop,
                'feature_class': feature_class,
                'feature_code': feature_code,
                'cols': cols,
            }


def in_extent(lat, lon, extent):
    (min_lat, min_lon), (max_lat, max_lon) = extent
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def filter_features_in_extent(txt_path, extent):
    matches = []
    for rec in load_geonames(txt_path):
        # keep only populated places (feature code PPL)
        if not rec.get('feature_code','').startswith('PPL'):
            continue
        if in_extent(rec['lat'], rec['lon'], extent):
            matches.append(rec)
    return matches


def main(txt_path, extent):
    matches = filter_features_in_extent(txt_path, extent)
    print(f'Found {len(matches)} features inside extent {extent}')
    # for m in matches[:20]:
    #     print(f"{m.get('asciiname','')}	{m['lat']}	{m['lon']}	pop={m.get('population',0)}")

    outlist = []
    for m in matches:
        # outlist.append(m)
        outlist.append({
            'asciiname': m.get('asciiname',''),
            'latitude': m.get('lat', None),
            'longitude': m.get('lon', None),
            'population': m.get('population', 0),
        })
    return outlist


if __name__ == '__main__':
    outlist = main("inputs/JP.txt", ((34.5, 139.0), (36.0, 141.0)))

    outjson = "japan_cities_in_extent.json"
    with open(outjson, 'w', encoding='utf-8') as jf:
        json.dump(outlist, jf, ensure_ascii=False, indent=2)
    print('Wrote', outjson)
