; Inno Setup script for the CandyCrunch Windows installer. Build the app with packaging/candycrunch.spec first, then:
;     iscc /DAppVersion=0.7.0 packaging\candycrunch.iss
; which writes dist\CandyCrunch-<version>-Windows-Setup.exe
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{4E1A620D-4376-4048-8BAC-5FA02B8EE921}
AppName=CandyCrunch
AppVersion={#AppVersion}
AppPublisher=Bojar Lab, University of Gothenburg
AppPublisherURL=https://github.com/BojarLab/CandyCrunch
DefaultDirName={autopf}\CandyCrunch
DefaultGroupName=CandyCrunch
DisableProgramGroupPage=yes
; Lab computers often lack administrator rights: by default the app installs for the current user only, an installation for all users stays possible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=CandyCrunch-{#AppVersion}-Windows-Setup
SetupIconFile=candycrunch.ico
UninstallDisplayIcon={app}\CandyCrunch.exe
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\CandyCrunch\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An update replaces the previous version's files instead of mixing with them
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\CandyCrunch"; Filename: "{app}\CandyCrunch.exe"
Name: "{autodesktop}\CandyCrunch"; Filename: "{app}\CandyCrunch.exe"; Tasks: desktopicon

[Registry]
; Saved results open in the app on double-click, and mzML, mzXML and mgf files offer "Open with CandyCrunch"
Root: HKA; Subkey: "Software\Classes\.candycrunch"; ValueType: string; ValueName: ""; ValueData: "CandyCrunch.Results"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\CandyCrunch.Results"; ValueType: string; ValueName: ""; ValueData: "CandyCrunch results"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\CandyCrunch.Results\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\CandyCrunch.exe,0"
Root: HKA; Subkey: "Software\Classes\CandyCrunch.Results\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\CandyCrunch.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\CandyCrunch.Spectra"; ValueType: string; ValueName: ""; ValueData: "LC-MS/MS run"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\CandyCrunch.Spectra\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\CandyCrunch.exe"" ""%1"""
Root: HKA; Subkey: "Software\Classes\.mzML\OpenWithProgids"; ValueType: string; ValueName: "CandyCrunch.Spectra"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.mzXML\OpenWithProgids"; ValueType: string; ValueName: "CandyCrunch.Spectra"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.mgf\OpenWithProgids"; ValueType: string; ValueName: "CandyCrunch.Spectra"; ValueData: ""; Flags: uninsdeletevalue

[Run]
Filename: "{app}\CandyCrunch.exe"; Description: "{cm:LaunchProgram,CandyCrunch}"; Flags: nowait postinstall skipifsilent
