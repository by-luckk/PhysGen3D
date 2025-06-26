// interactive_angle_drag.js

const ANGLE_SELECTOR_DEBUG_SHOW_FILENAME = false; // Set to true for debugging
const ANGLE_SELECTOR_CIRCLE_LINE_WIDTH_PX = 3; // Width of the circle outline
const ANGLE_SELECTOR_CIRCLE_RADIUS_PROPORTION = 0.39; // Proportion of image width for the circle radius
const ANGLE_SELECTOR_CIRCLE_CENTER_X_PROPORTION = 0.5; // Center X of circle (0.5 = image center)
const ANGLE_SELECTOR_CIRCLE_CENTER_Y_PROPORTION = 0.5; // Center Y of circle (0.5 = image center)
const ANGLE_SELECTOR_DRAG_START_TOLERANCE_PX = 30; // How close to circle edge to start drag

// Icon properties
const ICON_PATH = "static/images/icon.png";
let iconImage = null;
let iconLoaded = false;
// NEW: Scale proportion for the icon relative to the main image width
const ICON_SCALE_PROPORTION_OF_IMAGE_WIDTH = 0.12; // e.g., 10% of image width
let actualIconDisplayWidthPx = 40; // Default, will be calculated

document.addEventListener('DOMContentLoaded', () => {
    const demoContainerElements = document.querySelectorAll('.js-angle-drag-demo');
    demoContainerElements.forEach(containerElement => {
        initAngleSelectorInstance(containerElement);
    });
});

function initAngleSelectorInstance(containerElement) {
    let INITIAL_IMAGE_PATH;
    let VIDEOS_BASE_PATH;
    let VIDEO_DATA_PATH;

    const staticImage = containerElement.querySelector('.angle-static-image');
    const canvas = containerElement.querySelector('.angle-canvas-overlay');
    const videoPlayer = containerElement.querySelector('.angle-video-player');
    const debugFilenameDisplay = búsquedaAscendente(containerElement, '.debug-filename-display');

    if (!staticImage || !canvas || !videoPlayer) {
        console.error(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): Core child elements missing.`);
        if (debugFilenameDisplay && ANGLE_SELECTOR_DEBUG_SHOW_FILENAME) {
            debugFilenameDisplay.textContent = "Error: Core HTML elements missing.";
        }
        return;
    }

    const rootDir = containerElement.dataset.rootDir;
    if (!rootDir) {
        console.error(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): data-root-dir attribute missing.`);
        if (debugFilenameDisplay && ANGLE_SELECTOR_DEBUG_SHOW_FILENAME) {
            debugFilenameDisplay.textContent = "Error: data-root-dir missing.";
        }
        return;
    }

    const effectiveRootDir = rootDir.endsWith('/') ? rootDir : rootDir + '/';
    INITIAL_IMAGE_PATH = effectiveRootDir + (containerElement.dataset.initialImageFile || "initial_frame.png");
    VIDEOS_BASE_PATH = effectiveRootDir + (containerElement.dataset.videosDir || "videos/");
    VIDEO_DATA_PATH = effectiveRootDir + (containerElement.dataset.specsFile || "video_specs.json");

    let videoAngleData = {};
    let availableUserAngles = [];
    let imageWidth = 0, imageHeight = 0;
    let circleCenterX = 0, circleCenterY = 0, circleRadiusPx = 0;
    let currentBeadMathAngleRad = 0; // Math angle (radians, canvas coords) of the bead during drag
    let lastSelectedUserAngleDeg = 0.0; // Stores the "User Angle" (degrees) for the icon's state

    let isDragging = false;
    const ctx = canvas.getContext('2d');

    iconImage = new Image();
    iconImage.onload = () => {
        iconLoaded = true;
        console.log(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): Icon image loaded from ${ICON_PATH}`);
        if (canvas.style.display === 'block' && imageWidth > 0) {
            const mathAngleForRedraw = convertUserAngleToMathAngleRad(lastSelectedUserAngleDeg);
            drawCircleAndIcon(mathAngleForRedraw, -((lastSelectedUserAngleDeg - 180 + 360) % 360));
        }
    };
    iconImage.onerror = () => {
        console.error(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): Failed to load icon image from ${ICON_PATH}`);
        iconImage = null;
    };
    iconImage.src = ICON_PATH;

    function búsquedaAscendente(el, selector) {
        let ancestro = el.parentElement;
        while (ancestro) {
            if (ancestro.matches(selector)) return ancestro.querySelector('.debug-filename-display');
            if (ancestro.querySelector && ancestro.querySelector(selector)) return ancestro.querySelector(selector);
            ancestro = ancestro.parentElement;
        }
        const siblingOrChild = containerElement.parentElement?.querySelector(selector) || containerElement.querySelector(selector);
        if (siblingOrChild && siblingOrChild.classList.contains('debug-filename-display')) return siblingOrChild;
        return null;
    }

    function convertMathAngleToUserAngle(angleRadFromAtan2_CanvasY_Down) {
        let canvasAngleDeg = angleRadFromAtan2_CanvasY_Down * (180 / Math.PI);
        if (canvasAngleDeg < 0) {
            canvasAngleDeg += 360;
        }
        let standardMathAngleDeg;
        if (canvasAngleDeg === 0) {
            standardMathAngleDeg = 0;
        } else {
            standardMathAngleDeg = (360 - canvasAngleDeg) % 360;
        }
        return (standardMathAngleDeg + 180) % 360;
    }

    function convertUserAngleToMathAngleRad(userAngleDegFromJsonOrLastSelected) {
        let standardMathAngleDeg = (userAngleDegFromJsonOrLastSelected - 180 + 360) % 360;
        let canvasAngleDeg;
        if (standardMathAngleDeg === 0) {
            canvasAngleDeg = 0;
        } else {
            canvasAngleDeg = (360 - standardMathAngleDeg) % 360;
        }
        return canvasAngleDeg * (Math.PI / 180);
    }

    function findClosestUserAngle(targetUserAngleDeg, userAngleList) {
        if (!userAngleList || userAngleList.length === 0) return null;
        return userAngleList.reduce((prev, curr) => {
            const distPrev = Math.min(Math.abs(targetUserAngleDeg - prev), 360 - Math.abs(targetUserAngleDeg - prev));
            const distCurr = Math.min(Math.abs(targetUserAngleDeg - curr), 360 - Math.abs(targetUserAngleDeg - curr));
            return distCurr < distPrev ? curr : prev;
        });
    }

    function drawCircleAndIcon(positionalCanvasAngleRad, rotationDegreesToUse) {
        if (!imageWidth || !imageHeight || !circleRadiusPx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.arc(circleCenterX, circleCenterY, circleRadiusPx, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = ANGLE_SELECTOR_CIRCLE_LINE_WIDTH_PX;
        // --- MODIFICATION START ---
        ctx.setLineDash([5, 5]); // Set the line to be dashed (e.g., 5 pixels on, 5 pixels off)
        ctx.stroke();
        ctx.setLineDash([]); // Reset line dash to solid for any subsequent drawing
        // --- MODIFICATION END ---

        const iconX = circleCenterX + circleRadiusPx * Math.cos(positionalCanvasAngleRad);
        const iconY = circleCenterY + circleRadiusPx * Math.sin(positionalCanvasAngleRad);

        if (iconLoaded && iconImage) {
            ctx.save();
            ctx.translate(iconX, iconY);

            const rotationRadians = rotationDegreesToUse * (Math.PI / 180);
            ctx.rotate(rotationRadians);

            const iconDrawWidth = actualIconDisplayWidthPx;
            const iconAspectRatio = iconImage.naturalHeight / iconImage.naturalWidth;
            const iconDrawHeight = iconDrawWidth * iconAspectRatio;

            ctx.drawImage(iconImage, -iconDrawWidth / 2, -iconDrawHeight / 2, iconDrawWidth, iconDrawHeight);
            ctx.restore();
        } else {
            // Fallback if icon isn't loaded (draws a red dot)
            ctx.beginPath();
            ctx.arc(iconX, iconY, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
            ctx.fill();
        }

        if (ANGLE_SELECTOR_DEBUG_SHOW_FILENAME && debugFilenameDisplay) {
            const userAngleForDisplay = convertMathAngleToUserAngle(positionalCanvasAngleRad);
            debugFilenameDisplay.textContent = `Icon Pos (User Angle): ${userAngleForDisplay.toFixed(1)}°, Icon Rot (Direct): ${rotationDegreesToUse.toFixed(1)}°`;
        }
    }

    function switchToStaticImage(clearDebug = true) {
        videoPlayer.style.display = 'none';
        if (!videoPlayer.paused) videoPlayer.pause();

        staticImage.style.display = 'block';
        if (imageWidth > 0 && imageHeight > 0) {
            canvas.style.display = 'block';
            canvas.width = imageWidth;
            canvas.height = imageHeight;
            // lastSelectedUserAngleDeg now stores the precise user-dragged angle
            const mathAngleForRedraw = convertUserAngleToMathAngleRad(lastSelectedUserAngleDeg);
            const rotationForIcon = -((lastSelectedUserAngleDeg - 180 + 360) % 360);
            drawCircleAndIcon(mathAngleForRedraw, rotationForIcon);
        } else {
            canvas.style.display = 'none';
        }

        // The debug message is updated by drawCircleAndIcon, so no need to clear explicitly
        // unless a very specific "Select an angle" message is desired when clearDebug is true
        // and no angle has been truly set. For now, drawCircleAndIcon handles it.
    }

    function setupImageAndCanvas() {
        let currentOffsetWidth = staticImage.offsetWidth;
        let currentOffsetHeight = staticImage.offsetHeight;

        let determinedImageWidth = 0;
        let determinedImageHeight = 0;

        if (currentOffsetWidth > 0 && currentOffsetHeight > 0) {
            determinedImageWidth = currentOffsetWidth;
            determinedImageHeight = currentOffsetHeight;
        } else if (staticImage.naturalWidth > 0 && staticImage.naturalHeight > 0) {
            // Fallback to natural dimensions if offset dimensions are 0 (e.g., element is hidden)
            console.warn(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): offsetWidth/offsetHeight is 0. This can happen if the element or its parent is hidden (e.g., display: none). Falling back to naturalWidth/naturalHeight (${staticImage.naturalWidth}x${staticImage.naturalHeight}) for canvas sizing.`);
            determinedImageWidth = staticImage.naturalWidth;
            determinedImageHeight = staticImage.naturalHeight;
        }

        // Update global imageWidth/imageHeight which are used by other functions like drawCircleAndIcon and switchToStaticImage
        imageWidth = determinedImageWidth;
        imageHeight = determinedImageHeight;

        if (imageWidth > 0 && imageHeight > 0) {
            canvas.width = imageWidth;
            canvas.height = imageHeight;
            actualIconDisplayWidthPx = imageWidth * ICON_SCALE_PROPORTION_OF_IMAGE_WIDTH;
            circleRadiusPx = Math.min(imageWidth * ANGLE_SELECTOR_CIRCLE_RADIUS_PROPORTION, imageHeight * ANGLE_SELECTOR_CIRCLE_RADIUS_PROPORTION);
            circleCenterX = imageWidth * ANGLE_SELECTOR_CIRCLE_CENTER_X_PROPORTION;
            circleCenterY = imageHeight * ANGLE_SELECTOR_CIRCLE_CENTER_Y_PROPORTION;

            lastSelectedUserAngleDeg = 0.0; // Initial "User Angle"
            currentBeadMathAngleRad = convertUserAngleToMathAngleRad(lastSelectedUserAngleDeg); // Corresponding math angle

            console.log(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): Setup Canvas. Img (used for canvas): ${imageWidth}x${imageHeight}. Icon Width: ${actualIconDisplayWidthPx.toFixed(1)}px`);
        } else {
            // This is the original error condition if neither offset nor natural dimensions are valid
            console.error(`Angle Drag Demo (${containerElement.id}): Failed to get valid image dimensions (neither offset nor natural are > 0). Path: ${INITIAL_IMAGE_PATH}`); // This is line 206 in the original code.
            canvas.style.display = 'none';
            // imageWidth and imageHeight remain 0, switchToStaticImage will handle hiding canvas.
        }
        switchToStaticImage(false); // Draw initial state (or hide canvas if dimensions are invalid)
    }

    function handleVideoInteractionInitiation(e) {
        // Check if the video is currently being "served" (i.e., visible and playing)
        if (videoPlayer.style.display === 'block' && !videoPlayer.paused && videoPlayer.readyState >= videoPlayer.HAVE_CURRENT_DATA) {

            // Prevent default for touch events to avoid page scroll during the drag that's about to start.
            if (e.type.startsWith('touch')) {
                e.preventDefault(); 
            }

            console.log(`Angle Drag Demo (${containerElement.id || 'Unknown ID'}): Interaction on active video. Switching to static and initiating angle selection.`);

            // Switch to the static image view. The canvas overlay will become active.
            switchToStaticImage(); 

            // Now, call handleDragStart as if the mousedown/touchstart occurred directly on the canvas.
            // handleDragStart will use the event's coordinates to calculate the initial angle,
            // set isDragging = true, and add document listeners for move/end.
            handleDragStart(e);
        }
        // If the video is visible but paused, or not ready, this function does nothing on mousedown/touchstart.
        // The 'ended' listener and other interactions will handle those cases as before.
    }

    function handleDragStart(e) {
        // Check if canvas and circle parameters are initialized
        if (!imageWidth || !imageHeight || !circleRadiusPx) {
            console.warn(`Angle Drag Demo (${containerElement.id}): Drag start ignored, setup incomplete.`);
            // Attempt to refresh dimensions if function exists and setup is incomplete
            if (typeof containerElement.forceRefreshDimensions === 'function') {
                containerElement.forceRefreshDimensions();
                // Re-check if dimensions are now valid
                if (!imageWidth || !imageHeight || !circleRadiusPx) return;
            } else {
                return; // Exit if setup is still incomplete
            }
        }

        // If a video is playing, switch to the static image view first.
        // The icon will be at its lastSelectedUserAngleDeg from the previous interaction.
        if (videoPlayer.style.display === 'block' && !videoPlayer.paused) {
            switchToStaticImage();
        }

        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        const canvasX = clientX - rect.left;
        const canvasY = clientY - rect.top;

        // --- MODIFICATION FOR CLICK ANYWHERE ---
        // The drag will now start regardless of where the user clicks on the canvas.
        // The ANGLE_SELECTOR_DRAG_START_TOLERANCE_PX check is removed.

        isDragging = true;

        // Calculate the angle from the circle center to the actual click/touch point.
        // This angle determines the initial position of the bead on the circle,
        // effectively "snapping" the interaction point to the circle.
        currentBeadMathAngleRad = Math.atan2(canvasY - circleCenterY, canvasX - circleCenterX);
        
        // Update the icon's visual state immediately to reflect this new starting angle.
        const currentUserAngle = convertMathAngleToUserAngle(currentBeadMathAngleRad);
        const rotationForIcon = -((currentUserAngle - 180 + 360) % 360);
        drawCircleAndIcon(currentBeadMathAngleRad, rotationForIcon);

        // Add event listeners for drag movement and end.
        document.addEventListener('mousemove', handleDragMove);
        document.addEventListener('mouseup', handleDragEnd);
        document.addEventListener('touchmove', handleDragMove, { passive: false }); // passive: false to allow preventDefault
        document.addEventListener('touchend', handleDragEnd);
        document.addEventListener('touchcancel', handleDragEnd);
        
        // Prevent default browser actions (e.g., text selection during drag, page scrolling on touch).
        e.preventDefault();
        // --- END OF MODIFICATION ---
    }

    function handleDragMove(e) {
        if (!isDragging) return;

        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
            e.preventDefault();
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        const canvasX = clientX - rect.left;
        const canvasY = clientY - rect.top;

        currentBeadMathAngleRad = Math.atan2(canvasY - circleCenterY, canvasX - circleCenterX);
        const currentUserAngle = convertMathAngleToUserAngle(currentBeadMathAngleRad);
        const rotationForIcon = -((currentUserAngle - 180 + 360) % 360);
        drawCircleAndIcon(currentBeadMathAngleRad, rotationForIcon);
    }

    // MODIFIED handleDragEnd
    function handleDragEnd() {
        if (!isDragging) return;
        isDragging = false;

        document.removeEventListener('mousemove', handleDragMove);
        document.removeEventListener('mouseup', handleDragEnd);
        document.removeEventListener('touchmove', handleDragMove);
        document.removeEventListener('touchend', handleDragEnd);
        document.removeEventListener('touchcancel', handleDragEnd);

        // currentBeadMathAngleRad holds the precise math angle from the drag.
        // Convert it to our "User Angle" format. This is the actual angle the user dragged to.
        const actualUserAngle = convertMathAngleToUserAngle(currentBeadMathAngleRad);

        // Store this actual angle. This will be used by switchToStaticImage
        // to draw the icon at this precise location and orientation.
        lastSelectedUserAngleDeg = actualUserAngle;
        
        // Redraw the icon one last time at its final dragged position and orientation.
        // The positional angle is currentBeadMathAngleRad.
        // The rotation is derived from the actualUserAngle.
        const actualRotationForIcon = -((actualUserAngle - 180 + 360) % 360);
        drawCircleAndIcon(currentBeadMathAngleRad, actualRotationForIcon);

        // --- Video Selection Logic (still snaps to closest available video) ---
        let videoFilenameToPlay = null;
        let closestUserAngleForVideo = null;

        if (Object.keys(videoAngleData).length > 0 && availableUserAngles.length > 0) {
            closestUserAngleForVideo = findClosestUserAngle(actualUserAngle, availableUserAngles);

            if (closestUserAngleForVideo !== null) {
                videoFilenameToPlay = videoAngleData[closestUserAngleForVideo.toFixed(1)];
                console.log(`Icon resting at actual angle: ${actualUserAngle.toFixed(1)}°. Video selected for snapped angle: ${closestUserAngleForVideo.toFixed(1)}° -> ${videoFilenameToPlay || 'None available'}`);

                if (ANGLE_SELECTOR_DEBUG_SHOW_FILENAME && debugFilenameDisplay) {
                     // Update debug text after drawCircleAndIcon might have set it, to be more specific
                     debugFilenameDisplay.textContent = `Icon@${actualUserAngle.toFixed(1)}°, Vid@${closestUserAngleForVideo.toFixed(1)}°: ${videoFilenameToPlay || 'No file'}`;
                }

            } else {
                 console.log(`Icon resting at actual angle: ${actualUserAngle.toFixed(1)}°. No closest video angle found in JSON.`);
                 if (ANGLE_SELECTOR_DEBUG_SHOW_FILENAME && debugFilenameDisplay) {
                    debugFilenameDisplay.textContent = `Icon@${actualUserAngle.toFixed(1)}° (No video found)`;
                 }
            }
        } else {
            console.log(`Icon resting at actual angle: ${actualUserAngle.toFixed(1)}°. No video data loaded or no angles available.`);
             if (ANGLE_SELECTOR_DEBUG_SHOW_FILENAME && debugFilenameDisplay) {
                 debugFilenameDisplay.textContent = `Icon@${actualUserAngle.toFixed(1)}° (No video data)`;
             }
        }
        // --- End Video Selection Logic ---

        if (videoFilenameToPlay) {
            videoPlayer.src = VIDEOS_BASE_PATH + videoFilenameToPlay;
            const onVideoReadyToPlay = () => {
                videoPlayer.removeEventListener('loadeddata', onVideoReadyToPlay);
                videoPlayer.removeEventListener('error', onVideoLoadError);
                if (imageWidth > 0 && imageHeight > 0) {
                    videoPlayer.style.width = imageWidth + 'px';
                    videoPlayer.style.height = imageHeight + 'px';
                } else {
                    videoPlayer.style.width = '100%'; videoPlayer.style.height = 'auto';
                }
                staticImage.style.display = 'none';
                canvas.style.display = 'none'; // Hide canvas overlay during video playback
                videoPlayer.style.display = 'block';
                videoPlayer.play().catch(err => {
                    console.error(`Angle Drag Demo (${containerElement.id}): Error playing video ${videoFilenameToPlay}:`, err);
                    // On error, switch back. switchToStaticImage will use lastSelectedUserAngleDeg (the actual angle).
                    switchToStaticImage(false); 
                });
            };
            const onVideoLoadError = (ev) => {
                videoPlayer.removeEventListener('loadeddata', onVideoReadyToPlay);
                videoPlayer.removeEventListener('error', onVideoLoadError);
                console.error(`Angle Drag Demo (${containerElement.id}): Error loading video ${videoFilenameToPlay}:`, ev);
                switchToStaticImage(false); // Switch back, icon uses actual angle
            };
            videoPlayer.addEventListener('loadeddata', onVideoReadyToPlay);
            videoPlayer.addEventListener('error', onVideoLoadError);
            videoPlayer.load();
        } else {
            // No video to play. The icon is already drawn at the precise actualUserAngle.
            // Ensure UI is in static image mode.
            // switchToStaticImage will use lastSelectedUserAngleDeg (which is actualUserAngle).
            // Pass false to preserve any specific debug message set above.
            switchToStaticImage(false); 
        }
    }

    function initializeInteractiveDemo() {
        staticImage.src = INITIAL_IMAGE_PATH;
        staticImage.onload = () => {
            staticImage.onload = null; // Prevent re-triggering if src is set again
            // Corrected console.log statement:
            console.log(`Angle Drag Demo (${containerElement.id}): Image loaded (${staticImage.naturalWidth}x${staticImage.naturalHeight}). Path: ${INITIAL_IMAGE_PATH}`);
            if (staticImage.naturalWidth > 0 && staticImage.naturalHeight > 0) {
                setupImageAndCanvas(); // This will now use the fallback if needed
            } else {
                console.error(`Angle Drag Demo (${containerElement.id}): Image loaded but has zero natural dimensions. Path: ${INITIAL_IMAGE_PATH}`);
                // Potentially hide or show an error message on the canvas/containerElement
            }
        };
        staticImage.onerror = () => {
            console.error(`Angle Drag Demo (${containerElement.id}): Error loading initial image. Path: ${INITIAL_IMAGE_PATH}`);
            // Potentially hide or show an error message on the canvas/containerElement
        };

        // Ensure image onload/onerror are set before checking .complete if src might already be set
        if (staticImage.complete && staticImage.getAttribute('src') === INITIAL_IMAGE_PATH) {
            // console.log(`Angle Drag Demo (${containerElement.id}): Image already loaded, processing. Path: ${INITIAL_IMAGE_PATH}`);
            // Defer to ensure DOM layout might have a chance, though naturalWidth/Height check is primary for data.
            // The setupImageAndCanvas will handle if offsetWidth/Height are still 0.
             if (staticImage.naturalWidth > 0 && staticImage.naturalHeight > 0) {
                console.log(`Angle Drag Demo (${containerElement.id}): Image already complete with natural dimensions (${staticImage.naturalWidth}x${staticImage.naturalHeight}). Calling setupImageAndCanvas. Path: ${INITIAL_IMAGE_PATH}`);
                setTimeout(setupImageAndCanvas, 0);
            } else if (staticImage.naturalWidth === 0 && staticImage.naturalHeight === 0 && staticImage.complete) {
                 console.error(`Angle Drag Demo (${containerElement.id}): Image already complete but has zero natural dimensions. Path: ${INITIAL_IMAGE_PATH}`);
            } else {
                 console.log(`Angle Drag Demo (${containerElement.id}): Image already complete but natural dimensions not positive or src mismatch. Will rely on onload if src is reset by forceRefresh or similar. Path: ${INITIAL_IMAGE_PATH}`);
            }
        }


        // Listen for drag initiation on the canvas (when static image is shown)
        canvas.removeEventListener('mousedown', handleDragStart); // Ensure no double listeners
        canvas.addEventListener('mousedown', handleDragStart);
        canvas.removeEventListener('touchstart', handleDragStart); // Ensure no double listeners
        canvas.addEventListener('touchstart', handleDragStart, { passive: false });

        // Video player event listeners
        videoPlayer.addEventListener('ended', () => {
            switchToStaticImage();
        });
        videoPlayer.addEventListener('error', (e) => {
            console.error(`Angle Drag Demo (${containerElement.id}): Video player error.`, e);
            switchToStaticImage(); // Switch to static on error
        });

        videoPlayer.removeEventListener('mousedown', handleVideoInteractionInitiation);
        videoPlayer.addEventListener('mousedown', handleVideoInteractionInitiation);
        videoPlayer.removeEventListener('touchstart', handleVideoInteractionInitiation);
        videoPlayer.addEventListener('touchstart', handleVideoInteractionInitiation, { passive: false });
    }

    containerElement.forceRefreshDimensions = function() {
        console.log(`Angle Drag Demo (${containerElement.id}): forceRefreshDimensions called.`);
        if (!staticImage.getAttribute('src') || staticImage.getAttribute('src') !== INITIAL_IMAGE_PATH) {
            staticImage.src = INITIAL_IMAGE_PATH; // Reload if src is wrong/missing
        } else if (staticImage.complete && staticImage.naturalWidth > 0) {
            setupImageAndCanvas(); // If src is correct and image is loaded
        } else if (staticImage.getAttribute('src')) { // Src is set but maybe not loaded
            if (!staticImage.onload) { // Add onload if not already trying to load
                staticImage.onload = () => {
                    staticImage.onload = null;
                    if (staticImage.naturalWidth > 0 && staticImage.naturalHeight > 0) setupImageAndCanvas();
                };
            }
        } else { // No src attribute, set it
            staticImage.src = INITIAL_IMAGE_PATH;
        }
    };

    fetch(VIDEO_DATA_PATH)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status} while fetching ${VIDEO_DATA_PATH}`);
            return response.json();
        })
        .then(data => {
            videoAngleData = data;
            availableUserAngles = Object.keys(videoAngleData).map(parseFloat).sort((a, b) => a - b);
            if (availableUserAngles.length === 0) {
                console.warn(`Angle Drag Demo (${containerElement.id}): No angles found in ${VIDEO_DATA_PATH}. Videos might not play.`);
            }
            initializeInteractiveDemo();
        })
        .catch(error => {
            console.error(`Angle Drag Demo (${containerElement.id}): Error fetching video data ${VIDEO_DATA_PATH}:`, error);
            initializeInteractiveDemo(); // Initialize even if video data fails, drag will work without videos
        });
}