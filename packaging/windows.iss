#ifndef BundleDir
  #error BundleDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{A3E1E4D9-921F-4184-9D40-F60B2B94F676}
AppName=吉他谱视频提取工具
AppVersion={#AppVersion}
AppPublisher=y1ng-h0ng
AppPublisherURL=https://github.com/y1ng-h0ng/guitar-tab-extractor
DefaultDirName={localappdata}\Programs\GuitarTabExtractor
DefaultGroupName=吉他谱视频提取工具
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=no
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename=GuitarTabExtractor-{#AppVersion}-Setup-x64
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\GuitarTabExtractor.exe
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "zhcn"; MessagesFile: "compiler:Default.isl,ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\吉他谱视频提取工具"; Filename: "{app}\GuitarTabExtractor.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\吉他谱视频提取工具"; Filename: "{app}\GuitarTabExtractor.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\GuitarTabExtractor.exe"; Description: "启动吉他谱视频提取工具"; Flags: nowait postinstall skipifsilent unchecked
