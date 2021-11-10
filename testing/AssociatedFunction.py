from collections import defaultdict

from API.Condition import ConditionType
from API.FactoryService import FactoryServices
from Functions.ToolFunction import ToolFunction


class AssociatedFunction(ToolFunction):

    def __init__(self):
        super().__init__("associated", [
            AssociatedResources()
        ])

    def getHelp(self):
        return """
Helps resolve complex associations in the ft-ontology.
Use:
associated resource - to print a list of services and their associated mason resources.
        """

    def runWithArgs(self, ftonto):
        print("Cannot be called without args")
        print(self.getHelp())


class AssociatedResources(ToolFunction):

    def __init__(self):
        super().__init__("resources")

    def getHelp(self):
        return """
Prints a list of services and their associated mason resources. A Service and a resource are associated if a condition (pre or post) exists such that the service uses that condition and resource is mentioned in the condition.
        """

    @staticmethod
    def findAssociatedResources(ftonto):

        # Map a condition to a associated resource of the checker service e.g. the resource for a light_barrier check is the light barrier itself
        # actualParameter is a map indexed by the url-key (e.g. resource) mapping to a tuple containing the formal parameter and the value the parameter is set to (both ids)
        resourceMapping = {
            ConditionType.STATE_OF_RESOURCE: lambda x: x.actualParameter["resource"][1],
            ConditionType.CHECK_POSITION: lambda x: x.actualParameter["resource"][1],
            ConditionType.LIGHT_BARRIER: lambda x: x.actualParameter["lb"][1],
            ConditionType.CAPACITIVE_SENSOR: lambda x: x.actualParameter["cs"][1],
            ConditionType.BUSINESS_KEY_CHECK: lambda x: x.actualParameter["reader_name"][1]
        }

        resourcesByService = defaultdict(lambda: {"byPrecondition": [], "byPostcondition": []})
        serviceByID = FactoryServices(ftonto).getAllInstances()
        for instance in [instance for _, instance in serviceByID.items() if
                         instance.preconditions is not [] or instance.postconditions is not []]:

            # Sanity check - no unknown condition types
            if [pre for pre in instance.preconditions if pre.type is ConditionType.UNKNOWN] or \
                    [post for post in instance.postconditions if post.type is ConditionType.UNKNOWN]:
                raise ValueError(f"WARN: Condition has unknown type in instance {instance}")

            # Map conditions to resource and filter services that dont have any
            if instance.preconditions or instance.postconditions:
                pre_iri_list = AssociatedResources.generate_iri_list(resourceMapping, serviceByID,
                                                                     instance.preconditions)
                post_iri_list = AssociatedResources.generate_iri_list(resourceMapping, serviceByID,
                                                                      instance.postconditions)

                resourcesByService[instance.name]["byPrecondition"] = pre_iri_list
                resourcesByService[instance.name]["byPostcondition"] = post_iri_list

        # print('Services/Keys before reduction:', len(resourcesByService.keys()))
        # keys_to_drop = []
        # for key, value in resourcesByService.items():
        #     by_pre = value.get("byPrecondition")
        #     by_post = value.get("byPostcondition")
        #
        #     if len(by_pre) < 2 and len(by_post)<2:
        #         print(f'Service {key} dropped.')
        #         keys_to_drop.append(key)
        #
        # for key in keys_to_drop:
        #     resourcesByService.pop(key)
        #
        # print('Services/Keys after reduction:', len(resourcesByService.keys()))

        # test_keys = [
        #     "Service_VGR_Pick_Up_And_Transport_With_Resource_VGR_1_With_Start_DM_2_Sink_With_End_Human_Workstation",
        #     "Service_VGR_Pick_Up_And_Transport_With_Resource_VGR_1_With_Start_DM_2_Sink_With_End_Oven",
        #     "Service_VGR_Pick_Up_And_Transport_With_Resource_VGR_1_With_Start_DM_2_Sink_With_End_PM_1_Sink"]
        # resourcesByService = {your_key: resourcesByService[your_key] for your_key in test_keys}

        precondition_pairs = {}
        postcondition_pairs = {}

        for key, value in resourcesByService.items():
            by_pre = value.get("byPrecondition")
            by_post = value.get("byPostcondition")

            for c1 in by_pre:
                if c1 in precondition_pairs.keys():
                    for c2 in by_pre:
                        if c1 != c2 and c2 not in precondition_pairs[c1]:
                            precondition_pairs[c1].append(c2)
                else:
                    precondition_pairs[c1] = [c2 for c2 in by_pre if c2 != c1]

            for c1 in by_post:
                if c1 in postcondition_pairs.keys():
                    for c2 in by_post:
                        if c1 != c2 and c2 not in postcondition_pairs[c1]:
                            postcondition_pairs[c1].append(c2)
                else:
                    postcondition_pairs[c1] = [c2 for c2 in by_post if c2 != c1]

        resourcesByService = {
            "precondition_pairs": precondition_pairs,
            "postcondition_pairs": postcondition_pairs
        }

        return resourcesByService

    @staticmethod
    def generate_iri_list(resourceMapping, serviceByID, conditions):

        iri_list = [resourceMapping[condition.type](serviceByID[condition.checkerID]) for condition in
                    conditions]

        replacements = {'FTOnto.': 'FTOnto:'}
        replacements_exact_match = {
            'FTOnto:HBW_1': 'FTOnto:HBW_1_Crane_Jib',
            'FTOnto:VGR_1': 'FTOnto:HBW_1_Crane_Jib',
            'FTOnto:HBW_2': 'FTOnto:HBW_2_Crane_Jib',
            'FTOnto:VGR_2': 'FTOnto:HBW_2_Crane_Jib',
        }

        for key, value in replacements.items():
            iri_list = [iri.replace(key, value) for iri in iri_list]

        iri_list = [replacements_exact_match[iri] if iri in replacements_exact_match.keys() else iri for iri in
                    iri_list]

        return iri_list

    def runWithArgs(self, ftonto):
        resourcesByService = AssociatedResources.findAssociatedResources(ftonto)
        import json
        path_to_file = 'service_condition_pairs.json'
        with open(path_to_file, 'w') as outfile:
            json.dump(resourcesByService, outfile, sort_keys=True, indent=2)
