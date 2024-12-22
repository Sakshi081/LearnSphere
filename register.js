import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-analytics.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyA3HwfpVzegW1jGXyoH_TQ6JXmRdD7puaE",
  authDomain: "learnsphere-a1c95.firebaseapp.com",
  projectId: "learnsphere-a1c95",
  storageBucket: "learnsphere-a1c95.firebasestorage.app",
  messagingSenderId: "652736746414",
  appId: "1:652736746414:web:273c100900291e1696ae3f",
  measurementId: "G-457Z527KH9"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);