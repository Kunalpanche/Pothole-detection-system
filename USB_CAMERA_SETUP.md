# USB Camera Setup Guide (DroidCam / IVCam)

This guide explains how to use your Android or iOS phone as a high-quality, low-latency USB webcam for the Pothole Detection System.

This method replaces the previous Wi-Fi browser-based streaming (`/video` endpoint), which was prone to high latency and network instability.

## Prerequisites

1.  **A Smartphone** (Android or iOS)
2.  **A USB Cable** capable of data transfer (not just charging)
3.  **A PC** running Windows, macOS, or Linux.

---

## Step 1: Install Webcam Software

You need an app that turns your phone into an external camera device for your PC. We recommend **DroidCam** or **IVCam**.

### Option A: DroidCam (Recommended, Free)
1.  **Phone:** Download [DroidCam from the Google Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam) or App Store.
2.  **PC:** Download and install the [DroidCam PC Client](https://www.dev47apps.com/).

### Option B: IVCam (High Quality)
1.  **Phone:** Download [iVCam from the Google Play Store](https://play.google.com/store/apps/details?id=com.e2esoft.ivcam) or App Store.
2.  **PC:** Download and install the [iVCam PC Client](https://www.e2esoft.com/ivcam/).

---

## Step 2: Connect Phone via USB

Wired USB connections guarantee the lowest latency and highest framerate for the pothole detection model.

### For Android:
1.  **Enable Developer Options:** Go to `Settings` > `About phone` and tap `Build number` 7 times.
2.  **Enable USB Debugging:** Go to `Settings` > `System` (or Developer Options) and turn on **USB Debugging**.
3.  **Connect USB:** Plug your phone into your PC using the USB cable. Allow USB debugging if prompted on your phone.

### For iOS:
1.  Just install iTunes on your PC to ensure the correct Apple drivers are installed.
2.  Plug your iPhone to your PC via USB and trust the computer.

---

## Step 3: Start the Camera Feed

1.  Open the Camera App (DroidCam or iVCam) on your phone.
2.  Open the corresponding PC Client on your computer.
3.  In the PC Client, select **USB connection**.
4.  You should now see your phone's camera feed displayed in the PC Client window.

> **Note:** Keep the PC client running in the background. It creates a "virtual webcam" that the pothole detection engine connects to.

---

## Step 4: Use it in the Pothole Detection App

1.  Start the Pothole Detection System: `python app.py`
2.  Open the web interface in your browser: [http://localhost:5000](http://localhost:5000)
3.  Click on the **"USB webcam"** tab over the mode selection panel.
4.  Click **"Refresh"** to detect available cameras.
5.  Select the camera from the drop-down menu (usually "DroidCam Source" or "e2eSoft iVCam" which translates to an index like 0, 1, or 2).
6.  Click **"Set USB Camera"**.
7.  Click **"Start Detection"**. Your phone's feed will appear in the UI with low-latency pothole tracking applied!
