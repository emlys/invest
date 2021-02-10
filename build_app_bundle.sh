#!/bin/bash -ve
#
# This script is assumed to be executed from the project root.
#
# Arguments:
#  $1 = the path to the binary dir to package.
#  $2 = the path to the HTML documentation
#  $3 = the path to where the application bundle should be written.

# remove temp files that can get in the way
tempdir=`basename $2`
echo "tempdir:"
echo "$tempdir"
if [ -d "$tempdir" ]
then
    rm -rfd "$tempdir"
fi

# prepare a local temp dir for a filesystem
mkdir -p "$tempdir"

new_basename='Example'
_APPDIR="$3"
_MACOSDIR="$_APPDIR/Contents/MacOS"
_RESOURCEDIR="$_APPDIR/Contents/Resources"
_EXAMPLE_DIST_DIR="$_MACOSDIR/example_dist"
echo "APPDIR:"
echo $3
echo "$_APPDIR/Contents/MacOS"

mkdir -p "${3}/Contents/MacOS"
mkdir -p "${_RESOURCEDIR}"

cp -r "$1" "$_EXAMPLE_DIST_DIR"

new_command_file="$_MACOSDIR/$new_basename"
cp "invest.icns" "$_RESOURCEDIR/invest.icns"

new_plist_file="$_APPDIR/Contents/Info.plist"
cp "Info.plist" "$new_plist_file"

# replace the version and application name strings in the Info.plist file
sed -i '' "s|++NAME++|$new_basename|g" "$new_plist_file"
sed -i '' "s|++VERSION++|${1}|g" "$new_plist_file"

# This is the command that will launch the application.
echo '#!/bin/bash' > $new_command_file
echo '#' >> $new_command_file
echo '# the QT_MAC_WANTS_LAYER definition is supposed to have been set by the' >> $new_command_file
echo "# runtime hook, but doesn't seem to be working.  Setting it here allows the" >> $new_command_file
echo "# binary to run on OSX Big Sur." >> $new_command_file
echo 'QT_MAC_WANTS_LAYER=1 `dirname $0`/example_dist/example launch' >> $new_command_file
chmod a+x $new_command_file

