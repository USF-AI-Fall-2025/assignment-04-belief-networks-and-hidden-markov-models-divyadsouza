from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("KeyPresent", "Starts"),
        ("Starts", "Moves"),
    ])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery": ['Works', "Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas": ['Full', "Empty"]},
)

cpd_key = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
    state_names={"KeyPresent": ['yes', "no"]},
)

cpd_radio = TabularCPD(
    variable="Radio", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable="Ignition", variable_card=2,
    values=[[0.75, 0.01], [0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works', "Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts": ['yes', 'no'], "Ignition": [
        "Works", "Doesn't work"], "Gas": ['Full', "Empty"], "KeyPresent": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01], [0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no']}
)


def main():
    # Associating the parameters with the model structure
    car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas,
                       cpd_radio, cpd_battery, cpd_moves, cpd_key)

    car_infer = VariableElimination(car_model)

    # The probability the car moves given the radio turns on and the car starts
    print("The probability the car Moves given the Radio turns on and the car Starts")
    print(car_infer.query(variables=["Moves"], evidence={
          "Radio": "turns on", "Starts": "yes"}))

    # Given that the car will not move, what is the probability that the battery is not working?
    print("Given that the car will not move, the probability that the battery is not working = ", end="")
    q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(q1.values[q1.state_names["Battery"].index("Doesn't work")], "\n")

    # Given that the radio is not working, what is the probability that the car will not start?
    print("Given that the radio is not working, the probability that the car will not start = ", end="")
    q2 = car_infer.query(variables=["Starts"], evidence={
                         "Radio": "Doesn't turn on"})
    print(q2.values[q2.state_names["Starts"].index("no")], "\n")

    # Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    q3 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    print("Radio working regardless of gas if Battery works : ",
          q3.values[q3.state_names["Radio"].index("turns on")])
    q4 = car_infer.query(variables=["Radio"], evidence={
                         "Gas": "Full", "Battery": "Works"})
    print("Radio working if Gas Full and Battery works : ",
          q4.values[q4.state_names["Radio"].index("turns on")])
    print("Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?\n", q3 != q4, "\n")

    # Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
    q5 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q5_val = q5.values[q5.state_names["Ignition"].index("Doesn't work")]
    print("Ignition failing regardless of gas if car doesn't Move : ", q5_val)
    q6 = car_infer.query(variables=["Ignition"], evidence={
                         "Gas": "Empty", "Moves": "no"})
    q6_val = q6.values[q6.state_names["Ignition"].index("Doesn't work")]
    print("Ignition failing if Gas Empty and car doesn't Move : ", q6_val)

    eval = "MORE" if q6_val > q5_val else "LESS" if q6_val < q5_val else "EQUALLY"
    print("The ignition is **", eval,
          "** likely to have failed if the car does not move while it does not have gas in it", "\n")

    # What is the probability that the car starts if the radio works and it has gas in it?
    q7 = car_infer.query(variables=["Starts"], evidence={
                         "Radio": "turns on", "Gas": "Full"})
    print("The probability that the car starts if the radio works and it has gas in it :",
          q7.values[q7.state_names["Starts"].index("yes")], "\n")

    # The probability that the key is not present given that the car does not move
    q_key_given_no_move = car_infer.query(
        variables=["KeyPresent"], evidence={"Moves": "no"})
    p_no_key = q_key_given_no_move.values[q_key_given_no_move.state_names["KeyPresent"].index(
        "no")]
    print("The probability that the key is not present given that the car does not move :", p_no_key)


if __name__ == "__main__":
    main()
