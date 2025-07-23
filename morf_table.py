#vstacking tables outputted form morf
#probs dont need a whole script for this so maybe make a useful funcs

from astropy.table import Table, vstack
import os

def stack_mfmtk_outputs(output_dir, output_file=None, file_format='ascii.csv'):
    """
    Stacks all .mfmtk output tables from a directory into one Astropy Table,
    after removing the last column and the 'A1Sersic' column from each table.

    Parameters:
        output_dir (str): Directory containing .mfmtk files.
        output_file (str or None): Optional output file path (e.g. .fits or .csv).
        file_format (str): Format for reading the tables (default: 'ascii.csv').

    Returns:
        Table: Combined Astropy Table from all .mfmtk files.
    """
    import os
    from astropy.table import Table, vstack

    mfmtk_files = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.mfmtk')
    ])

    tables = []
    for filepath in mfmtk_files:
        try:
            table = Table.read(filepath, format=file_format)

            # Remove the last column (likely problematic)
            if len(table.colnames) > 0:
                table.remove_column(table.colnames[-1])

            # Remove the problematic 'A1Sersic' column if it exists
            if 'A1Sersic' in table.colnames:
                table.remove_column('A1Sersic')

            # Unmask and clean metadata
            if table.masked:
                table = table.filled()

            # Strip units and metadata to avoid merge issues
            for col in table.colnames:
                table[col].unit = None
                table[col].description = None
                table[col].format = None

            tables.append(table)

        except Exception as e:
            print(f"❌ Could not read {filepath}: {e}")

    if not tables:
        raise ValueError("No .mfmtk files could be read successfully.")

    try:
        combined_table = vstack(tables, join_type='exact')
    except Exception as e:
        print(f"⚠️ Exact stacking failed: {e}")
        print("🔁 Retrying with join_type='outer'")
        combined_table = vstack(tables, join_type='outer')

    # Sort by ID column (replace 'ID' with your actual ID column name)
    if '# rootname9.65' in combined_table.colnames:
        combined_table.sort('# rootname9.65')
    else:
        print("⚠️ Warning: No 'ID' column found to sort by.")

    if output_file:
        combined_table.write(output_file, overwrite=True)
        print(f"✅ Combined table saved to: {output_file}")

    return combined_table


# === Call the function ===
combined_table = stack_mfmtk_outputs(
    output_dir='/raid/scratch/work/Griley/GALFIND_WORK/Cutouts/v13/JADES-DR3-GS-East/ACS_WFC+NIRCam/1.50as/F444W/data',
    output_file='combined_morfometryka.fits',
    file_format='ascii.csv'
)