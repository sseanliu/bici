// MediaPipe Hand Tracking Module
// Detects hand landmarks, renders hand skeleton, and detects pinch gestures

class HandTracker {
  constructor() {
    this.handLandmarker = null;
    this.isTracking = false;
    this.lastDetectionTime = -1;

    // Hand landmarks connections for skeleton rendering
    this.HAND_CONNECTIONS = [
      [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],        // Index finger
      [0, 9], [9, 10], [10, 11], [11, 12],   // Middle finger
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [5, 9], [9, 13], [13, 17]              // Palm
    ];

    // Pinch state
    this.isPinching = false;
    this.pinchPosition = { x: 0, y: 0 };
    this.pinchThreshold = 0.05; // Distance threshold for pinch detection

    // Callbacks
    this.onPinchStart = null;
    this.onPinchMove = null;
    this.onPinchEnd = null;
  }

  async init() {
    try {
      // Dynamically import MediaPipe from CDN
      const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
      const { FilesetResolver, HandLandmarker } = vision;

      // Load WASM files
      const visionResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      // Create hand landmarker
      this.handLandmarker = await HandLandmarker.createFromOptions(visionResolver, {
        baseOptions: {
          modelAssetPath: "/core/models/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      console.log('Hand tracker initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize hand tracker:', error);
      return false;
    }
  }

  detectHands(videoElement, timestamp) {
    if (!this.handLandmarker || !this.isTracking) return null;

    // Only detect if timestamp has changed
    if (timestamp === this.lastDetectionTime) return null;
    this.lastDetectionTime = timestamp;

    try {
      const results = this.handLandmarker.detectForVideo(videoElement, timestamp);

      // Process pinch detection
      if (results.landmarks && results.landmarks.length > 0) {
        this.processPinchGesture(results.landmarks[0], results.handednesses[0]);
      } else {
        // No hands detected - end pinch if active
        if (this.isPinching) {
          this.isPinching = false;
          if (this.onPinchEnd) this.onPinchEnd();
        }
      }

      return results;
    } catch (error) {
      console.error('Hand detection error:', error);
      return null;
    }
  }

  processPinchGesture(landmarks, handedness) {
    // Get thumb tip (landmark 4) and index finger tip (landmark 8)
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];

    // Calculate Euclidean distance
    const distance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) +
      Math.pow(thumbTip.y - indexTip.y, 2) +
      Math.pow(thumbTip.z - indexTip.z, 2)
    );

    // Get midpoint between thumb and index for pinch position
    const pinchX = (thumbTip.x + indexTip.x) / 2;
    const pinchY = (thumbTip.y + indexTip.y) / 2;

    const wasPinching = this.isPinching;
    this.isPinching = distance < this.pinchThreshold;

    if (this.isPinching) {
      this.pinchPosition = { x: pinchX, y: pinchY };

      if (!wasPinching) {
        // Pinch just started
        if (this.onPinchStart) this.onPinchStart(pinchX, pinchY);
      } else {
        // Pinch continuing - moving
        if (this.onPinchMove) this.onPinchMove(pinchX, pinchY);
      }
    } else if (wasPinching) {
      // Pinch just ended
      if (this.onPinchEnd) this.onPinchEnd();
    }
  }

  renderHandSkeleton(ctx, landmarks, canvasWidth, canvasHeight) {
    if (!landmarks || landmarks.length === 0) return;

    // Save current context state
    ctx.save();

    // Draw connections (skeleton)
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (const [start, end] of this.HAND_CONNECTIONS) {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];

      const x1 = startPoint.x * canvasWidth;
      const y1 = startPoint.y * canvasHeight;
      const x2 = endPoint.x * canvasWidth;
      const y2 = endPoint.y * canvasHeight;

      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }

    ctx.stroke();

    // Draw landmark points
    ctx.fillStyle = '#FF0000';
    for (const landmark of landmarks) {
      const x = landmark.x * canvasWidth;
      const y = landmark.y * canvasHeight;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Highlight thumb and index finger tips for pinch visualization
    if (landmarks.length > 8) {
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];

      // Draw larger circles for pinch points
      ctx.fillStyle = this.isPinching ? '#FFFF00' : '#00FFFF';

      ctx.beginPath();
      ctx.arc(thumbTip.x * canvasWidth, thumbTip.y * canvasHeight, 8, 0, 2 * Math.PI);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(indexTip.x * canvasWidth, indexTip.y * canvasHeight, 8, 0, 2 * Math.PI);
      ctx.fill();

      // Draw line between thumb and index
      ctx.strokeStyle = this.isPinching ? '#FFFF00' : '#00FFFF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(thumbTip.x * canvasWidth, thumbTip.y * canvasHeight);
      ctx.lineTo(indexTip.x * canvasWidth, indexTip.y * canvasHeight);
      ctx.stroke();
    }

    // Restore context state
    ctx.restore();
  }

  renderAllHands(ctx, results, canvasWidth, canvasHeight) {
    if (!results || !results.landmarks) return;

    for (let i = 0; i < results.landmarks.length; i++) {
      this.renderHandSkeleton(ctx, results.landmarks[i], canvasWidth, canvasHeight);
    }
  }

  start() {
    this.isTracking = true;
  }

  stop() {
    this.isTracking = false;
    if (this.isPinching && this.onPinchEnd) {
      this.onPinchEnd();
    }
    this.isPinching = false;
  }

  getPinchState() {
    return {
      isPinching: this.isPinching,
      position: this.pinchPosition
    };
  }
}
