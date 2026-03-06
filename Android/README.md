## 모바일프로그래밍 과제/실습용 안드로이드 프로젝트

- 루트 프로젝트 이름: `MobileProgramming`
- 주차별 모듈: `ch01`, `ch02`, `ch03` (각각 별도 안드로이드 앱 모듈)
- 언어: **Java**
- DB: **내장 SQLite** (안드로이드 기본 제공, 별도 설치 불필요)

### 1. 안드로이드 스튜디오에서 여는 방법
1. 안드로이드 스튜디오 실행
2. `Open` 또는 `Open an Existing Project` 선택
3. 폴더 선택:  
   `c:\Users\kwakm\OneDrive\Desktop\Cusor-Project\Dongyang\26-2-1\Android`
4. Gradle Sync 완료 후, 왼쪽 `Project` 뷰에서 **Android** 모드로 변경
5. `ch01`, `ch02`, `ch03` 중 원하는 모듈을 실행 앱으로 설정해서 사용

### 2. 요구 환경
- Android Studio (최신 버전 권장)
- Android SDK (Android 15 / API 35 근처 버전 설치 권장)
- JDK 17 (Android Studio 번들 JDK 사용 권장)

### 3. SQLite 사용
- 안드로이드에는 이미 SQLite가 포함되어 있어서 **별도 설치는 필요 없음**
- 과제/실습에서 사용할 때는 `SQLiteOpenHelper` 또는 Room 등으로 접근
- 필요 시 각 `ch0x` 모듈 안에 DB 관련 패키지/클래스를 추가해서 사용

### 4. 모듈 구조 개요
- `ch01/` : 1주차 예제/실습용 앱
- `ch02/` : 2주차 예제/실습용 앱
- `ch03/` : 3주차 예제/실습용 앱

각 모듈에는 다음과 같은 기본 구조가 생성됨:
- `src/main/AndroidManifest.xml`
- `src/main/java/.../MainActivity.java`
- `src/main/res/layout/activity_main.xml`
- `src/main/res/values/strings.xml`

