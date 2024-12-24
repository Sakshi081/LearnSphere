import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-analytics.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "xxxxxxxxxxxxxxxx",
  authDomain: "xxxxxxxxxxxxxxxxxxxx",
  projectId: "xxxxxxxxxxxxxxx",
  storageBucket: "xxxxxxxxxxxxx",
  messagingSenderId: "xxxxxxxxxxx",
  appId: "xxxxxxxxxxxxx",
  measurementId: "xxxxxxxxxx"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
