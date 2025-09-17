FlowSense: An Intelligent Traffic Management System

An intelligent, adaptive traffic management system that leverages computer vision to optimize traffic flow, reduce congestion, and enhance urban mobility.
📖 Overview

Welcome to FlowSense, a smart traffic management solution designed for modern cities. Traditional traffic light systems operate on fixed timers, leading to inefficient traffic flow and unnecessary congestion. FlowSense addresses this problem by using real-time video feeds from CCTV cameras to analyze traffic density and dynamically adjust signal timings. By employing the powerful YOLOv8 object detection model, the system accurately identifies and counts vehicles, making intelligent decisions to keep traffic moving smoothly.
🌟 Key Features

    Real-time Traffic Analysis: Utilizes the YOLOv8 model for high-accuracy vehicle detection and classification on live video streams.

    Adaptive Signal Control: Intelligently adjusts traffic signal durations based on the real-time vehicle count and density at intersections.

    Intuitive Admin Dashboard: A user-friendly graphical user interface (GUI) built with Tkinter for monitoring live camera feeds, viewing traffic statistics, and manually overriding signals if needed.

    Simulation Mode: Includes a simulation environment to demonstrate the system's effectiveness and test algorithms without needing physical cameras.

    Modular & Scalable: Designed with a modular architecture that allows for easy expansion to more complex, multi-intersection networks.

🛠️ How It Works

The system operates in a continuous loop to manage traffic dynamically.

    Video Capture: The system captures live video streams from CCTV cameras positioned at a traffic intersection.

    Vehicle Detection: Each frame from the video is processed by the YOLOv8 model, which detects and categorizes vehicles (cars, trucks, buses, motorcycles).

    Density Calculation: The number of detected vehicles in each lane is counted to determine the current traffic density.

    Signal Optimization: A control algorithm analyzes the density data and calculates the optimal green-light duration for each lane to maximize traffic throughput.

    GUI Visualization: The Admin Dashboard displays the live feed with bounding boxes around detected vehicles, shows real-time vehicle counts, and visualizes the current signal status.

💻 Technology Stack

    Programming Language: Python

    Computer Vision: OpenCV

    Object Detection: YOLOv8

    GUI: Tkinter

    Dependencies: NumPy, Pillow

🚀 Getting Started

Follow these instructions to get a local copy of the project up and running.
Prerequisites

Make sure you have Python 3.8 or higher installed on your system.
Installation

    Clone the Repository:

    git clone [https://github.com/itsjayagovindk/FlowSense-Traffic-Management-System.git](https://github.com/itsjayagovindk/FlowSense-Traffic-Management-System.git)
    cd FlowSense-Traffic-Management-System

    Set up a Virtual Environment (Recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    Install Dependencies:
    Install all the required packages using the requirements.txt file.

    pip install -r requirements.txt

Running the Application

    Execute the main script to launch the Admin GUI and start the traffic management system:

    python main.py

🕹️ Usage

    Upon launching, the Admin Dashboard will open.

    Select the video source or camera feed you wish to monitor.

    The system will automatically begin analyzing the feed and controlling the virtual traffic signals.

    Observe the real-time vehicle counts and signal changes on the dashboard.

🔮 Future Enhancements

    Multi-Intersection Synchronization: Developing a centralized system to coordinate signals across multiple intersections.

    Emergency Vehicle Detection: Training the model to recognize and prioritize emergency vehicles (ambulances, fire trucks).

    Web-Based Dashboard: Migrating the Tkinter GUI to a modern web framework like Flask or Django for remote access.

    Predictive Analysis: Incorporating machine learning to predict traffic patterns based on historical data.
