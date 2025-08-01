document.addEventListener("DOMContentLoaded", function () {
    const video = document.querySelector("#video-feed");

    // Handle errors
    function handleError(error) {
        console.error("Error accessing webcam:", error);
        alert(
            "Error accessing webcam. Please ensure you have given camera permissions."
        );
    }
});
