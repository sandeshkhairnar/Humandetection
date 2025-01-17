plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.humandetect"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.humandetect"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}

dependencies {
    // AndroidX Camera
    implementation("androidx.camera:camera-core:1.1.0")
    implementation("androidx.camera:camera-camera2:1.1.0")
    implementation("androidx.camera:camera-lifecycle:1.1.0")
    implementation("androidx.camera:camera-view:1.1.0")
    implementation("com.google.android.material:material:1.5.0")


    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.8.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.2")

    // AndroidX AppCompat
    implementation("androidx.appcompat:appcompat:1.4.1")

    // AndroidX Core KTX
    implementation("androidx.core:core-ktx:1.7.0")

    // AndroidX ConstraintLayout (if you're using ConstraintLayout in your XML)
    implementation("androidx.constraintlayout:constraintlayout:2.1.3")
    implementation(libs.monitor)
    implementation(libs.ext.junit)
    testImplementation("junit:junit:4.12")
}