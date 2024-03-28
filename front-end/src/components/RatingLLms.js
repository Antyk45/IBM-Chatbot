import React, { useState, useEffect } from 'react';
import { FaStar } from 'react-icons/fa'; // Import the star icon
import './Rating.css'

function RatingLLMs({ llm, onRatingChange }) {
  const [rating, setRating] = useState(null);
  const [hover, setHover] = useState(null);

  const handleRatingChange = (newRating) => {
    setRating(newRating);
    onRatingChange(llm, newRating);
  };

  return (
    <div>
      {[...Array(5)].map((star, index) => {
        const currentRating = index + 1;
        return (
          <label key={index}>
            <input
              type="radio"
              name={`rating-${llm}`} // Unique name for each LLMS
              value={currentRating}
              onClick={() => handleRatingChange(currentRating)}
            />
            <FaStar
              className="stars"
              size={20}
              color={currentRating <= (hover || rating) ? "#ffc107" : "#e4e5e9"}
              onMouseEnter={() => setHover(currentRating)}
              onMouseLeave={() => setHover(null)}
            />
          </label>
        );
      })}
      <p>Your rating for {llm} is: {rating}</p>
    </div>
  );
}

export default RatingLLMs;
