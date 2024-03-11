// Assuming inclusion of Vector2D operations and definitions

class EvoMouse : public EvoFFNAnimat {
public:
    EvoMouse(): cheesesFound(0), totalVelocityMagnitude(0.0), velocityMeasurements(0) {
        // Constructor initialization as needed
    }

    // Method to update velocity tracking
    void UpdateVelocity(const Vector2D& velocity) {
        totalVelocityMagnitude += velocity.Magnitude(); // Assuming Vector2D has a Magnitude method
        velocityMeasurements++;
    }

    virtual float GetFitness() const {
        const float PowerPenalty = 0.1f;
        const float SpeedFactor = 1.0f;
        const float OptimalSpeed = (minSpeed + maxSpeed) / 2.0;

        // Calculate average speed
        float averageSpeed = velocityMeasurements > 0 ? totalVelocityMagnitude / velocityMeasurements : 0;
        float averageSpeedEfficiency = 1 - std::abs((averageSpeed - OptimalSpeed) / (maxSpeed - minSpeed));

        float fitness = static_cast<float>(cheesesFound) - (PowerPenalty * This.PowerUsed) + (SpeedFactor * averageSpeedEfficiency);

        return fitness > 0 ? fitness : 0; // Ensure fitness is non-negative
    }

private:
    int cheesesFound; // Updated as cheeses are collected
    float totalVelocityMagnitude; // Sum of all velocity magnitudes
    int velocityMeasurements; // Count of velocity updates for average calculation
    // Plus other members as required...
};
