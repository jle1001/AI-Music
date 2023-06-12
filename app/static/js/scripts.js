
function openFeature(evt, featureName) {
    // TODO: Documentate this function
    var i, featuresContent, featuresTab, featuresDesc;
    featuresContent = document.getElementsByClassName("features-content");
    featuresDesc = document.getElementsByClassName("features-description");
    for (i = 0; i < featuresContent.length; i++) {
        featuresContent[i].style.display = "none";
        featuresDesc[i].style.display = "none";
    }
    featuresTab = document.getElementsByClassName("features-tab");
    for (i = 0; i < featuresTab.length; i++) {
        featuresTab[i].className = featuresTab[i].className.replace(" active", "");
    }
    document.getElementById(featureName).style.display = "block";
    document.getElementById(featureName + "Desc").style.display = "block";
    evt.currentTarget.className += " active";
}

// Get the element with id="defaultOpen" and click on it
document.getElementById("defaultOpen").click();