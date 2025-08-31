# Information about each condition
condition_info = {
    "Frozen Teats": {
        "description": """
        Frozen teats is a condition that occurs when cow teats are exposed to extreme cold temperatures, 
        causing tissue damage similar to frostbite in humans. The teats can become hard, stiff, and painful, 
        making milk extraction difficult and potentially leading to more serious complications.
        """,
        "impact": """
        - **Pain and Discomfort**: Affected cows experience significant pain
        - **Reduced Milk Production**: Inability to properly milk the cow
        - **Secondary Infections**: Damaged tissue is susceptible to bacterial infections
        - **Permanent Damage**: In severe cases, affected tissue may die and slough off
        - **Economic Loss**: Due to reduced milk production and treatment costs
        """
    },
    "Healthy": {
        "description": """
        Healthy udders have uniform texture, appropriate size for the cow's breed and age, 
        and no visible abnormalities. The teats are properly formed, without cuts, cracks, or discoloration.
        Healthy udders are essential for optimal milk production and quality.
        """,
        "impact": """
        - **Optimal Milk Production**: Healthy udders result in maximum milk yield
        - **High Milk Quality**: Minimal risk of contamination or high somatic cell counts
        - **Economic Benefit**: Lower veterinary costs and higher milk production value
        - **Animal Welfare**: Improved comfort and wellbeing for the cow
        - **Longevity**: Cows with healthy udders typically remain productive longer
        """
    },
    "High Udder Score": {
        "description": """
        Udder scoring is a method to evaluate udder health and conformation. 
        A high udder score indicates excellent attachment, balance, and teat placement.
        Udders with high scores are well-attached to the body wall, have strong central ligament support,
        and feature properly placed teats.
        """,
        "impact": """
        - **Superior Milk Production**: Well-formed udders contribute to higher milk yield
        - **Reduced Injury Risk**: Better attachment reduces risk of udder injury
        - **Easier Milking**: Properly placed teats make milking more efficient
        - **Longevity**: Cows with high udder scores typically have longer productive lives
        - **Higher Value**: Cows with excellent udders often have higher dairy value
        """
    },
    "Medium Udder Score": {
        "description": """
        A medium udder score indicates acceptable but not ideal udder conformation.
        These udders may show slight imbalance, moderately strong attachment, 
        and acceptable teat placement with minor deviations from ideal.
        """,
        "impact": """
        - **Adequate Milk Production**: Generally supports good milk production
        - **Moderate Risk**: Some risk of developing udder health issues over time
        - **Management Attention**: May require additional monitoring
        - **Average Longevity**: Expected to provide standard productive lifespan
        - **Breeding Consideration**: May indicate need for improved udder traits in breeding
        """
    },
    "Low Udder Score": {
        "description": """
        A low udder score indicates poor udder conformation. These udders typically show
        weak attachment, poor balance between front and rear quarters, weak central ligament support,
        and/or poorly placed teats (too close together, too far apart, or pointing outward).
        """,
        "impact": """
        - **Reduced Milk Production**: Poor udder structure can limit milk yield
        - **Higher Risk of Injury**: Weakly attached udders are more susceptible to injury
        - **Milking Difficulties**: Poor teat placement can make milking challenging
        - **Shorter Productive Life**: Cows with poor udders often leave the herd earlier
        - **Increased Health Problems**: Higher risk of mastitis and other udder conditions
        """
    },
    "Mastitis": {
        "description": """
        Mastitis is an inflammation of the mammary gland and udder tissue, typically caused by bacterial infection.
        It's one of the most common and costly diseases in dairy cattle. Mastitis can be clinical (showing visible symptoms)
        or subclinical (detectable only through testing).
        """,
        "impact": """
        - **Reduced Milk Production**: Significant decrease in milk yield
        - **Poor Milk Quality**: Increased somatic cell count, altered composition
        - **Pain and Discomfort**: Causes suffering to affected animals
        - **Economic Loss**: Treatment costs, discarded milk, potential culling
        - **Antibiotic Use**: Treatment often requires antibiotics, raising concerns about resistance
        """
    },
    "Teat Lesions": {
        "description": """
        Teat lesions refer to physical damage to the teat surface, such as cuts, cracks, abrasions, or warts.
        These lesions can be caused by improper milking techniques, environmental factors, 
        chemical irritation, or infectious agents like the bovine papillomavirus (causing teat warts).
        """,
        "impact": """
        - **Pain During Milking**: Causes discomfort and stress to the cow
        - **Infection Risk**: Creates entry points for mastitis-causing bacteria
        - **Milking Challenges**: May interfere with proper milking procedures
        - **Decreased Milk Let-Down**: Pain can inhibit the milk ejection reflex
        - **Spread to Other Cows**: Some lesions (like warts) can be contagious
        """
    }
}

# Treatment information for each condition
treatment_info = {
    "Frozen Teats": {
        "immediate": """
        1. **Warming**: Gently warm the affected teats with warm (not hot) water
        2. **Dry Carefully**: Pat dry with clean, soft towels
        3. **Specialized Creams**: Apply petroleum jelly or specialized frostbite creams
        4. **Pain Management**: Administer pain medication as prescribed by a veterinarian
        5. **Gentle Handling**: Milk gently to avoid further tissue damage
        """,
        "management": """
        1. **Protected Milking Area**: Ensure milking is done in a sheltered, warm environment
        2. **Regular Application**: Continue applying protective ointments
        3. **Modified Milking Schedule**: More frequent but gentler milking sessions
        4. **Monitoring**: Watch for signs of secondary infection
        5. **Documentation**: Keep records of treatment and recovery progress
        """,
        "prevention": """
        1. **Shelter Improvements**: Provide adequate wind breaks and shelter
        2. **Bedding Management**: Ensure dry, insulating bedding in cold weather
        3. **Protective Barriers**: Apply teat dips with emollients before cold exposure
        4. **Cold Weather Protocols**: Implement specific procedures for extreme weather
        5. **Facility Design**: Consider barn layout to minimize drafts and cold spots
        """
    },
    "Healthy": {
        "immediate": """
        1. **Maintain Current Practices**: Continue good hygiene and management
        2. **Regular Monitoring**: Perform routine udder examinations
        3. **Proper Milking**: Maintain optimal milking procedures and equipment settings
        4. **Documentation**: Keep records of udder health status
        5. **Quality Testing**: Continue regular milk quality testing
        """,
        "management": """
        1. **Nutrition**: Ensure balanced diet with adequate vitamins and minerals
        2. **Hydration**: Provide clean, fresh water at all times
        3. **Stress Reduction**: Maintain comfortable environment and handling
        4. **Bedding Quality**: Keep clean, dry bedding to prevent contamination
        5. **Regular Cleaning**: Maintain sanitary conditions in housing areas
        """,
        "prevention": """
        1. **Pre and Post Dipping**: Use effective teat dips consistently
        2. **Equipment Maintenance**: Regular check and service of milking equipment
        3. **Staff Training**: Ensure proper milking techniques are used
        4. **Fly Control**: Implement measures to reduce flies that can irritate teats
        5. **Selective Breeding**: Consider udder health traits in breeding decisions
        """
    },
    "High Udder Score": {
        "immediate": """
        1. **Maintain Best Practices**: Continue excellent management and milking procedures
        2. **Documentation**: Record conformation details for breeding decisions
        3. **Regular Evaluation**: Continue periodic udder scoring assessments
        4. **Proper Support**: Ensure udder support is maintained for older cows
        5. **Gentle Handling**: Continue careful udder preparation and handling
        """,
        "management": """
        1. **Selective Breeding**: Utilize this cow's genetics in breeding program
        2. **Optimal Nutrition**: Maintain excellent nutritional management
        3. **Body Condition Monitoring**: Keep cows at ideal body condition score
        4. **Comfort Focus**: Provide comfortable bedding and lying areas
        5. **Milking Routine**: Maintain consistent milking times and procedures
        """,
        "prevention": """
        1. **Breeding Program**: Continue selection for superior udder traits
        2. **Heifer Development**: Focus on proper growth rates for developing heifers
        3. **Transition Management**: Careful management of periparturient period
        4. **Injury Prevention**: Eliminate potential hazards in housing
        5. **Regular Assessment**: Continue periodic scoring to track any changes
        """
    },
    "Medium Udder Score": {
        "immediate": """
        1. **Regular Monitoring**: Implement more frequent udder assessment
        2. **Proper Milking**: Ensure correct milking machine settings
        3. **Complete Milking**: Verify complete milk-out at each milking
        4. **Documentation**: Record current status as baseline for comparison
        5. **Specialized Care**: Consider individualized milking procedures if needed
        """,
        "management": """
        1. **Nutritional Support**: Ensure balanced diet with emphasis on udder health
        2. **Additional Comfort**: Provide extra bedding for udder support
        3. **Stress Reduction**: Minimize environmental and handling stress
        4. **Regular Cleaning**: Maintain excellent hygiene standards
        5. **Body Condition**: Maintain optimal body condition score
        """,
        "prevention": """
        1. **Breeding Strategy**: Select for improved udder conformation in next generation
        2. **Preventive Care**: Implement rigorous mastitis prevention program
        3. **Equipment Checks**: More frequent checks of milking equipment
        4. **Staff Training**: Ensure proper handling and milking techniques
        5. **Housing Design**: Consider facility modifications to prevent udder injury
        """
    },
    "Low Udder Score": {
        "immediate": """
        1. **Veterinary Assessment**: Have a veterinarian evaluate the udder
        2. **Modified Milking**: Adjust milking equipment settings for gentle milking
        3. **Support Methods**: Consider udder support options if available
        4. **Careful Handling**: Implement extra-gentle udder preparation
        5. **Record Keeping**: Document specific conformation issues
        """,
        "management": """
        1. **Individual Monitoring**: Implement close monitoring schedule
        2. **Special Housing**: Consider special bedding or housing arrangements
        3. **Milking Order**: Milk affected cows at optimal times
        4. **Nutrition Review**: Assess and adjust diet for udder health support
        5. **Culling Consideration**: Evaluate economic impact and welfare concerns
        """,
        "prevention": """
        1. **Breeding Decisions**: Do not breed for replacement from these cows
        2. **Genetic Selection**: Use sires with strong udder conformation traits
        3. **Heifer Development**: Ensure proper growth rates for developing heifers
        4. **Early Intervention**: Address minor issues before they become severe
        5. **Herd Management**: Consider overall herd udder health strategy
        """
    },
    "Mastitis": {
        "immediate": """
        1. **Sample Collection**: Collect milk sample for culture before treatment
        2. **Antibiotic Therapy**: Administer appropriate antibiotics as prescribed
        3. **Frequent Milking**: Increase milking frequency to remove infected milk
        4. **Anti-inflammatory**: Provide anti-inflammatory medication for pain/swelling
        5. **Isolation**: Separate affected cows to prevent spread
        """,
        "management": """
        1. **Track Treatment**: Keep detailed records of treatments and response
        2. **Milk Disposal**: Properly discard milk during treatment and withdrawal
        3. **Testing**: Conduct follow-up testing to confirm recovery
        4. **Supportive Care**: Ensure hydration and nutrition during recovery
        5. **Environmental Management**: Improve bedding and housing cleanliness
        """,
        "prevention": """
        1. **Milking Hygiene**: Implement strict pre- and post-milking procedures
        2. **Proper Equipment**: Maintain and regularly check milking equipment
        3. **Dry Cow Therapy**: Consider appropriate dry cow antibiotic treatment
        4. **Vaccination**: Implement mastitis vaccination program if appropriate
        5. **Genetic Selection**: Select for mastitis resistance in breeding program
        """
    },
    "Teat Lesions": {
        "immediate": """
        1. **Gentle Cleaning**: Clean affected areas with mild antiseptic solution
        2. **Protective Barrier**: Apply teat sealants or barriers to protect wounds
        3. **Topical Treatment**: Use appropriate ointments for specific lesion types
        4. **Pain Management**: Provide pain relief if prescribed by veterinarian
        5. **Modified Milking**: Adjust milking equipment or hand-milk if necessary
        """,
        "management": """
        1. **Daily Inspection**: Check affected teats daily for signs of healing
        2. **Continued Treatment**: Apply prescribed treatments consistently
        3. **Equipment Adjustment**: Modify milking equipment settings if needed
        4. **Special Handling**: Implement careful udder preparation techniques
        5. **Isolation**: Separate cows with contagious lesions (like warts)
        """,
        "prevention": """
        1. **Equipment Maintenance**: Regularly check and maintain milking equipment
        2. **Proper Teat Dips**: Use high-quality pre- and post-milking teat dips
        3. **Environmental Management**: Eliminate sharp edges in housing and alleys
        4. **Weather Protection**: Provide protection from extreme weather conditions
        5. **Staff Training**: Train milking staff on proper techniques and lesion identification
        """
    }
}
