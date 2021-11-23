#event_lines, argument_lines, roles_lines

# to add option for less detailed checks

def check_event_trigger(ref_string, pred_string):
    return (ref_string == pred_string)
    pass

def check_event_type(ref_string, pred_string, event_lines):
    if pred_string in event_lines:
        return (ref_string == pred_string)
    else:
        print("invalid prediction")
        return False
    pass

def check_event_argument(ref_string, pred_string):
    return (ref_string == pred_string)
    pass

def check_argument_type(ref_string, pred_string, argument_lines):
    if pred_string in argument_lines:
        return (ref_string == pred_string)
    else:
        print("invalid prediction")
        return False
    pass

def check_argument_role(ref_string, pred_string, roles_lines):
    if pred_string in roles_lines:
        return (ref_string == pred_string)
    else:
        print("invalid prediction")
        return False
    pass

def calculate_f1(ref_lines, pred_lines, event_lines, argument_lines, roles_lines):
    
    total_preds = 0
    total_correct = 0
    predicted = 0   
    ground_truths = 0
    full_corrects = 0

    list_of_tracking_metrics = ['predicted_tuples',
                                'ground_truth_tuples',
                                'correct_predictions',
                                'events_count',
                                'correct_events',
                                'correct_event_type',
                                'correct_arguments',
                                'correct_argment_types',
                                'correct_argument_roles'
                                ]

    metric_counts = dict.fromkeys(list_of_tracking_metrics, 0)
    

    for i in range(0, min(len(ref_lines), len(pred_lines))):
        
        ref_line = ref_lines[i].strip()
        pred_line = pred_lines[i].strip()

        ref_tuples = ref_line.split('|')
        pred_tuples = pred_line.split('|')

        # find a way to compare multiple tuples

        # correct - t1 | t2 | t3
        # pred    - p1 | p2
        # postives = 3 [number of ground truths minus nones]
        # predicted_pos = 2 [number of preds minus nones]
        # TP = correct preds 
        # TP + FP = predicted
        # TP + FN = positives 
        # Precision = correct / predicted_pos 
        # Recall = correct / positives
        # f = pr/p+r

        # handling repeated predictions 
        # set_of_preds = set()
        # for pred_tuple in pred_tuples:
        #     set_of_preds.add(pred_tuple.strip())
        # pred_tuples = list(set_of_preds)

        for pred_tuple in pred_tuples:
            pred_strings = pred_tuple.split(';')

            # in the case of no argument detection, we only calculate the event trigger scores
            if(pred_strings[2].strip()) == 'None':
                max_matches = 0
                part_matches = []

                for ref_tuple in ref_tuples:
                    # ssss
                    ev1, ev2 = cal_f1_for_pair(ref_tuple, pred_tuple, event_lines)

                    pair_score = ev1+ev2

                    if pair_score > max_matches:
                        max_matches = pair_score
                        part_matches = (ev1, ev2)
                        pass
                    pass

                metric_counts['events_count'] += 1
                if ev1 == 1:
                    metric_counts['correct_events'] += 1
                if ev2 == 1:
                    metric_counts['correct_event_type'] += 1

                continue
            
            max_matches = 0
            part_matches = []

            for ref_tuple in ref_tuples:
                res = cal_f1_for_tuple(ref_tuple, pred_tuple, event_lines, argument_lines, roles_lines)

                tuple_score = sum(res)

                if tuple_score > max_matches:
                    max_matches = tuple_score
                    part_matches = res
                    pass
                pass

            metric_counts['predicted_tuples'] += 1
            metric_counts['events_count'] += 1

            if max_matches == 5:
                metric_counts['correct_predictions'] += 1
            if part_matches[0] == 1:
                metric_counts['correct_events'] += 1
            if part_matches[1] == 1:
                metric_counts['correct_event_type'] += 1
            if part_matches[2] == 1:
                metric_counts['correct_arguments'] += 1
            if part_matches[3] == 1:
                metric_counts['correct_argment_types'] += 1
            if part_matches[4] == 1:
                metric_counts['correct_argument_roles'] += 1
            pass
        
        for ref_tuple in ref_tuples:
            if(ref_tuple.split(';')[2].strip()) != 'None':
                metric_counts['ground_truth_tuples'] += 1
                ground_truths += 1

        pass
    
    print(metric_counts)
    # 
    #

def cal_f1_for_pair(ref_tuple: str ,
                    pred_tuple: str,
                    event_lines: list
                    ) -> list:
    
    ref_strings = ref_tuple.split(';')
    pred_strings = pred_tuple.split(';')

    ev1 = int( check_event_trigger(ref_strings[0].strip(), pred_strings[0].strip()) )
    ev2 = int( check_event_type(ref_strings[1].strip(), pred_strings[1].strip(), event_lines) )

    return ev1, ev2

def cal_f1_for_tuple(ref_tuple: str ,
                     pred_tuple: str,
                     event_lines: list,
                     argument_lines: list,
                     roles_lines: list
                     ) -> list:

    ref_strings = ref_tuple.split(';')
    pred_strings = pred_tuple.split(';')

    ev1 = int( check_event_trigger(ref_strings[0].strip(), pred_strings[0].strip()) )
    ev2 = int( check_event_type(ref_strings[1].strip(), pred_strings[1].strip(), event_lines) )
    ev3 = int( check_event_argument(ref_strings[2].strip(), pred_strings[2].strip()) )
    ev4 = int( check_argument_type(ref_strings[3].strip(), pred_strings[3].strip(), argument_lines) )
    ev5 = int( check_argument_role(ref_strings[4].strip(), pred_strings[4].strip(), roles_lines) )

    ret = [ev1, ev2, ev3, ev4, ev5]
    
    return ret
