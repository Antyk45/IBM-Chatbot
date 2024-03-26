import React, { useState } from 'react';
import { saveAs } from 'file-saver';
import './InsertDocument.css'; // Import CSS file for styling

/**
 * This function handles the file input needed for the user to add their own csv files. 
 * @returns It returns the file the user has inputted for use in the chatbot. It also has some text in the return as this is how react structures text
 */
function FileUpload() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    // Save the file locally (in the 'uploads' folder)
    const reader = new FileReader();
    reader.onload = (e) => {
      const fileContent = e.target.result;
      const fileName = 'my-file.csv'; // Set your desired file name
      const blob = new Blob([fileContent], { type: 'text/csv' });
      saveAs(blob, fileName); // Save the file using file-saver
    };
    reader.readAsArrayBuffer(file);
  };

  return (
    <div className="file-upload-container">
    <div className="upload-text">
       <h1>Welcome to the Insert Document page!</h1>
       <h2>Just click the button below to insert the .csv file of your choice.</h2>
       <h3>Then head on over to the Chat With Me tab and ask as many questions as you want!</h3>
    </div>   
   <input type="file" accept=".csv" onChange={handleFileChange} />
      <button className="upload-button" onClick={() => document.querySelector('input[type="file"]').click()}>
        Upload CSV
      </button>
    </div>
  );
}

export default FileUpload;
