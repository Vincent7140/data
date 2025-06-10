def download_zipfile(url, archive_path, unzip_path, md5_hash):
    local_archive = os.path.join("local_downloads", os.path.basename(archive_path))
    if os.path.exists(local_archive):
        print(f"Using local archive for {os.path.basename(archive_path)}")
        shutil.copy(local_archive, archive_path)
    else:
        print(f"ERROR: {local_archive} not found. Please download it manually.")
        sys.exit(1)

    with zipfile.ZipFile(archive_path, "r") as fid:
        fid.extractall(unzip_path)
