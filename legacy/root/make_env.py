with open(".env", "r") as f:
    lines = f.readlines()

with open("env.yaml", "w") as out:
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"").strip("\'")
            out.write(f'{key}: "{val}"\n')

print("Successfully generated env.yaml with string values")
