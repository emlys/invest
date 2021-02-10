## Minimal example for PyInstaller bug report

The build is run in github actions following the steps in `.github/workflows/binary-applications.yml`.

To reproduce the problem:
1. Download the latest DMG build artifact from this branch
2. Open the DMG. (may have to right-click > open > open anyway to get around unidentified developer warning)
3. Open the Console app.
3. Open the example app that's in the DMG (should be a small empty window with the title "Example Launcher")
4. Search on "system policy: deny" in the Console app and see that there are several denied attempts to read user preferences:
```
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.VoiceMemos/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.archiveutility/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.Home/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.Safari/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.iChat/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.CloudDocs/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.mail/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.news/Data/Library/Preferences
Sandbox: example(55,805) deny(1) file-read-data /Users/emily/Library/Containers/com.apple.stocks/Data/Library/Preferences
```

This does not happen if the Qt event loop doesn't start (if you take out `APP.exec_()` from `launcher.py`).
