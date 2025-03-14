document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("imageInput");
    const detectButton = document.getElementById("detectButton");
    const imagePreview = document.getElementById("imagePreview");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const resultContainer = document.getElementById("resultContainer");
    const diseaseName = document.getElementById("diseaseName");
    const confidenceLevel = document.getElementById("confidenceLevel");
    const treatmentInfo = document.getElementById("treatmentInfo");

    // ✅ Show Image Preview
    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // ✅ Handle Image Upload & Disease Detection
    detectButton.addEventListener("click", async function () {
        if (!fileInput.files.length) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);

        // Show Loading Indicator
        resultContainer.classList.add("hidden");
        loadingIndicator.classList.remove("hidden");

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            console.log("API Response:", data);

            // ✅ Ensure Data is Valid
            if (data.disease && data.confidence !== undefined && data.treatment) {
                diseaseName.innerText = data.disease;
                confidenceLevel.innerText = data.confidence.toFixed(2);
                treatmentInfo.innerText = data.treatment;

                resultContainer.classList.remove("hidden");
            } else {
                alert("Error: Invalid API response");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Failed to fetch prediction. Check the backend.");
        } finally {
            loadingIndicator.classList.add("hidden");
        }
    });
});
