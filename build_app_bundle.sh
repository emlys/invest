#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.

# Make the app bundle
mkdir -p "build/mac_app/Example.app/Contents/MacOS"
cp -r "dist/example" "build/mac_app/Example.app/Contents/MacOS/example_dist"
cp "Info.plist" "build/mac_app/Example.app/Contents/Info.plist"


new_command_file="build/mac_app/Example.app/Contents/MacOS/Example"
# This is the command that will launch the application.
echo '#!/bin/bash' > $new_command_file
echo '#' >> $new_command_file
echo '# the QT_MAC_WANTS_LAYER definition is supposed to have been set by the' >> $new_command_file
echo "# runtime hook, but doesn't seem to be working.  Setting it here allows the" >> $new_command_file
echo "# binary to run on OSX Big Sur." >> $new_command_file
echo 'QT_MAC_WANTS_LAYER=1 `dirname $0`/example_dist/example launch' >> $new_command_file
chmod a+x $new_command_file

