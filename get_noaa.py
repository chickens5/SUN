import json
import pathlib
import time
import urllib.request


NOAA = {
    "solarwind": "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://services.swpc.noaa.gov/json/geospace/geospace_dst_7_day.json", #Measures Disturbance Storm Time (Dst) index measures the strength of geomagnetic storms driven by the solar wind interacting with Earth's magnetic field
}

#HTTP headers to identify as a browser (some NOAA endpoints reject urllib.request default)
HEADERS = {
    "User-Agent": "(https://github.com/chickens5/SUN, gabe3jackson@gmail.com)"
}

#propogated-solar-wind.json has known columns: ["time_tag", "speed", "density", "temperature", "bx", "by", "bz", "bt", "vx", "vy", "vz", "propagated_time_tag"]
#Thus, the following scheme splits the consolidated solarwind endpoint

PLASMA_COLS = ["time_tag", "speed", "density", "temperature", "vx", "vy", "vz"]
MAG_COLS = ["time_tag", "bx", "by", "bz", "bt"]

#Fetches a URL (NOAA JSON endpoints) with retry/timeout logic.
def fetch_json(url: str, timeout: int = 30, retries: int = 3) -> list | dict:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
            return json.loads(payload)
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            # Backoff: 0.5s on first retry, 1.0s on second
            time.sleep(0.5 * attempt)
    
    raise RuntimeError(f"Fetch failed after {retries} attempts for {url}: {last_error}")

#Breaks the propogated-solar-wind.json into a tuple of two lists for 1. Solar Wind Plasma & 2. Interplanetary Magnetic Flux (IMF) 
def split_consolidated_response(solarwind_raw: list) -> tuple[list, list]:
    """
    Parameters
    ----------
    solarwind_raw : list
        Raw response from propagated-solar-wind.json
        Format: [[header_row], [data_row_1], [data_row_2], ...]
        Header: ["time_tag", "speed", "density", "temperature", "bx", "by", "bz", "bt", "vx", "vy", "vz", "propagated_time_tag"]
    
    Returns
    -------
    tuple[list, list]
        (plasma_array, mag_array) where each is in same format as input
        plasma_array columns: ["time_tag", "speed", "density", "temperature", "vx", "vy", "vz"]
        mag_array columns: ["time_tag", "bx", "by", "bz", "bt"]
    """
    if not solarwind_raw or len(solarwind_raw) < 1:
        raise ValueError("Empty solarwind response")
    
    header = solarwind_raw[0]
    data_rows = solarwind_raw[1:]
    
    #Maps column names to their indices
    col_to_idx = {col: idx for idx, col in enumerate(header)}
    
    #Extracts indices for plasma and mag columns
    plasma_indices = [col_to_idx[col] for col in PLASMA_COLS if col in col_to_idx]
    mag_indices = [col_to_idx[col] for col in MAG_COLS if col in col_to_idx]
    
    if not plasma_indices or not mag_indices:
        raise ValueError(f"Missing required columns. Header: {header}")
    
    #Constructs the plasma and mag rows
    plasma_rows = [[row[i] for i in plasma_indices] for row in data_rows]
    mag_rows = [[row[i] for i in mag_indices] for row in data_rows]
    
    #Reconstructs the arrays with their column names
    plasma_header = [PLASMA_COLS[i] for i in range(len(PLASMA_COLS))]
    mag_header = [MAG_COLS[i] for i in range(len(MAG_COLS))]
    
    plasma_array = [plasma_header] + plasma_rows
    mag_array = [mag_header] + mag_rows
    
    return plasma_array, mag_array

#Saves the fetched data as JSON
def save_json(filename: str, data: list | dict, output_dir: str = "data") -> pathlib.Path:
    """
    Parameters
    ----------
    filename : str
        Output filename (e.g., "plasma.json")
    data : list | dict
        Data to save
    output_dir : str
        Output directory (default: "data"); created if doesn't exist
        
    Returns
    -------
    pathlib.Path
        Path to saved file
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {filename} ({len(data) if isinstance(data, list) else '?'} rows)")
    return file_path

#Fetches JSON endpoint, splits the raw solarwind response, and saves JSON locally.
def main() -> None:
    """Fetch all NOAA endpoints, split consolidated data, save locally."""
    print("Fetching NOAA SWPC data...")
    
    #Fetches solarwind
    print("\n1. Fetching propagated solar wind (consolidated)...")
    solarwind_raw = fetch_json(NOAA["solarwind"])
    print(f"Got {len(solarwind_raw)} rows")
    
    #Splits into plasma and mag
    print("\n2. Splitting into plasma and magnetic components...")
    plasma_array, mag_array = split_consolidated_response(solarwind_raw)
    print(f"   Plasma: {len(plasma_array)} rows")
    print(f"   Mag:    {len(mag_array)} rows")
    
    #Fetches Kp separately
    print("\n3. Fetching Kp index...")
    kp_raw = fetch_json(NOAA["kp"])
    print(f"Got {len(kp_raw) if isinstance(kp_raw, list) else len(kp_raw.get('data', []))} rows")
    
    
    #Fetches Dst (optional)
    print("\n5. Fetching Dst...")
    try:
        dst_raw = fetch_json(NOAA["dst"])
        print(f"Got {len(dst_raw)} rows")
    except Exception as e:
        print(f"Error...Skipped: {e}")
        dst_raw = None
    
    #Saves all to disk
    print("\n6. Saving to local JSON files...")
    save_json("plasma.json", plasma_array)
    save_json("mag.json", mag_array)
    save_json("kp.json", kp_raw)
   
    if dst_raw is not None:
        save_json("dst.json", dst_raw)
    
    print("\nSuccess! All data fetched and saved to data/ folder")


if __name__ == "__main__":
    main()
