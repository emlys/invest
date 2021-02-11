#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.

# Make the app bundle
mkdir -p "build/mac_app/ExampleApp.app/Contents/MacOS"
cp -r "dist/example_" "build/mac_app/ExampleApp.app/Contents/MacOS/"
cp "Info.plist" "build/mac_app/ExampleApp.app/Contents/Info.plist"


executable_file="build/mac_app/ExampleApp.app/Contents/MacOS/ExampleApp"
# This is the command that will launch the application.
echo '#!/bin/bash' > $executable_file
echo '#' >> $executable_file
echo '# the QT_MAC_WANTS_LAYER definition is supposed to have been set by the' >> $executable_file
echo "# runtime hook, but doesn't seem to be working.  Setting it here allows the" >> $executable_file
echo "# binary to run on OSX Big Sur." >> $executable_file
echo 'QT_MAC_WANTS_LAYER=1 `dirname $0`/example_ launch' >> $executable_file
chmod a+x $executable_file

