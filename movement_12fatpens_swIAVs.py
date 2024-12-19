"""
Python add-on for data preprocessing and trade movement process in the metapop
"""

import csv
import dateutil.parser             as     dup
import numpy                       as     np
import random

from   emulsion.agent.managers     import MetapopProcessManager
from   emulsion.tools.preprocessor import EmulsionPreprocessor
from   emulsion.tools.debug        import debuginfo
from   emulsion.model.exceptions   import SemanticException
from   emulsion.tools.functions    import random_normal, random_gamma
from   betapert                    import pert, mpert

#===============================================================
# Preprocessor class for restructuring the file of trade movements
#===============================================================
class TradeMovementsReader(EmulsionPreprocessor):
    """A preprocessor class for reading the CSV that describes the trade
    movements and restructuring it as a dictionary, stored in shared
    information in the simulation.

    """
    def init_preprocessor(self):
        if self.input_files is None or 'trade_file' not in self.input_files:
            raise SemanticException("A valid 'trafe_file' must be specified in the input_files section for pre-processing class {}".format(self.__class__.__name__))

    def run_preprocessor(self):
        """Expected file format: CSV with following fields

        - date: date of the movement (ISO format)
        - source: ID of the source herd
        - dest: ID of the dest herd
        - age: age group of the animals to move
        - quantity: amount of animals to move
        """
        if 'moves' not in self.simulation.shared_data:
            debuginfo('Reading {}'.format(self.input_files.trade_file))
            self.simulation.shared_data['moves'] = self.restructure_moves()
        else:
            debuginfo('Trade movements already loaded in simulation')

    def restructure_moves(self):
        """Restructure the CSV file as nested dictionaries"""

        # read a CSV data file for moves:
        # date of movement, source herd, destination herd, age group, quantity

        # and restructure it according to origin_date and delta_t:
        # {step: {source_id: [(dest_id, age_group, dur), ...],
        #         ...},
        #  ...}
        origin = self.model.origin_date
        step_duration = self.model.step_duration
        moves = {}
        with open(self.input_files.trade_file) as csvfile:
            # read the CSV file
            csvreader = csv.DictReader(csvfile, delimiter=',')
            for row in csvreader:
                week = dup.parse(row['date'])
                if week < origin:
                    # ignore dates before origin_date
                    continue
                # convert dates into simulation steps
                step = (week - origin) // step_duration
                # group information by step and source herd
                if step not in moves:
                    moves[step] = {}
                src, dest, dur = int(row['source']), int(row['dest']), float(row['duration'])
                if src not in moves[step]:
                    moves[step][src] = []
                moves[step][src].append([dest, dur])
        return moves

#===============================================================
# CLASS Metapopulation (LEVEL 'metapop')
#===============================================================
class Metapopulation(MetapopProcessManager):
    """
    level of the metapop.

    """

    #----------------------------------------------------------------
    # Processes
    #----------------------------------------------------------------
    def farmer_movement(self):
        """
        herd 0 is for non-gestating gilts and sows (A)
        herd 1 is for gestating gilts and sows (G)
        herd 2 is for farrowing (F) pigs and newborn piglets (Jnb)
        herd 3 is for weaned piglets in the nursery (Jn)
        herd 4-15 is for growing/fattening pigs (Jf)
        herd 16 is for dressing room 1 (no pigs)
        herd 17 is for dressing room 2 (no pigs)
        """
        moves = self.simulation.shared_data['moves']
        herds = self.get_populations()
        
        ## TRANSFER PIGS TO THEIR ASSIGNED GROUP
        
        # record gestating sows/gilt from nongestating group
        gestating_sows = herds[0].select_atoms('age_group', 'G')
        # remove gestating sows/gilt from nongestating group
        herds[0].remove_atoms(gestating_sows)
        # add gestating sows/gilt to gestating group
        herds[1].add_atoms(gestating_sows)
        
        # record farrowing sows from gestating group
        farrowing_sows = herds[1].select_atoms('age_group', 'F')
        # record number of farrowing sows to determine number of newborns
        herds[1].statevars.nb_new_farrowing_sows = len(farrowing_sows)
        # remove farrowing sows from gestating group
        herds[1].remove_atoms(farrowing_sows)
        # add farrowing sows to farrowing group
        herds[2].add_atoms(farrowing_sows)
        
        vert_trans = self.model.parameters.vert_trans
        # produce newborns
        newborn = []
        qty = self.model.parameters.pba
        for sow in farrowing_sows:
            # number of newborns must be less than capacity
            if herds[2].total_Jnb < self.model.parameters.K_herd:
                # produced infected newborns
                if (sow.is_in_state('I') or sow.is_in_state('E')) and vert_trans > random.random():
                    newborn_prototype = 'newborn_E'
                # susceptible sows produce susceptible newborns
                elif sow.is_in_state('S'):
                    newborn_prototype = 'newborn_S'
                # produce newborns with maternal immunity
                else:
                    newborn_prototype = 'newborn_M'
                mean_pba = round(self.model.parameters.mean_pba)
                sd_pba = round(self.model.parameters.sd_pba)
                pba = round(random.gauss(mean_pba, sd_pba))
                newborn = newborn + [herds[2].new_atom(sublevel='animals', prototype=newborn_prototype) for _ in range(pba)]
                
        #newborn = [herds[2].new_atom(sublevel='animals', prototype='newborn_M') 
        #           if vert_trans < np.random.rand()
        #           else herds[2].new_atom(sublevel='animals', prototype='newborn_E')
        #           for _ in range(qty)]
        if len(newborn) != 0:
            herds[2].add_atoms(newborn)
        #herds[1].add_atoms(piglets)
        
        # transfer corresponding age group of pigs to the correct herd
        nongestating_sows = herds[2].select_atoms('age_group', 'A')
        herds[2].remove_atoms(nongestating_sows)
        herds[0].add_atoms(nongestating_sows)
        
        nursery = herds[2].select_atoms('age_group', 'Jn')
        herds[2].remove_atoms(nursery)
        herds[3].add_atoms(nursery)
        
        growers = herds[3].select_atoms('age_group', 'Jf')
        herds[3].remove_atoms(growers) 

        # random testing for infectious or exposed animals and then removed from the population
        growers = [pig for pig in growers if not (pig.is_in_state('I') and np.random.rand() < self.model.parameters.proba_removal_if_I)]
        growers = [pig for pig in growers if not (pig.is_in_state('E') and np.random.rand() < self.model.parameters.proba_removal_if_E)]

        # counts and print the number of infected + exposed pigs
        #count_I_E = sum(pig.is_in_state('I') or pig.is_in_state('E') for pig in growers)
        #print(count_I_E)
            
        excess = len(growers) - 144       
        
        # only 12 fattening pigs per pen (12 pens), other are sold
        new_gilts = []
        if excess > 0:
            excess_growers = random.sample(growers, excess)
            # some fatteners are bred to gilts
            total_sows = herds[0].total_A + herds[1].total_G + herds[2].total_F
            if total_sows < 3 * self.model.parameters.K_sows: # multiplied by 3 to get total population of sows and gilts
                new_gilts = [pig.clone(prototype='nongestating') 
                             for pig in excess_growers if pig.is_in_state('Female')]
            # only 144 fatteners allowed
            growers = [x for x in growers if x not in excess_growers]
        
        # add new gilts to nongestating group
        if len(new_gilts) != 0:
            herds[0].add_atoms(new_gilts)
        
        if len(growers) > 0:
            # Specify the number of parts (n)
            n = np.floor(len(growers) / 12)
            if n == 0:
                n = 1
            elif n > 12:
                n = 12
        else: n = 1
        
        # Use array_split to split the array into n parts
        pens = np.array_split(growers, n)
        
        #transfer pigs to the fattening pens
        for i, pen in enumerate(pens):
            herds[i + 4].add_atoms(pen)
        
        ## MOVEMENT OF FARMERS

        # trans rate between groups goes back to zero per iteration
        for i in range(self.model.parameters.nb_herds):
            herds[i].statevars.trans_btwn_pens_frm_movement = 0
            
        if self.statevars.step in moves:
            trans_I_from_source = 0
            for source in moves[self.statevars.step]:
                for dest, dur in moves[self.statevars.step][source]:
                    if source != dest: 
                        #parameters for between pen transmission
                        # print(source, dest)
                        sd_bp_trans = (self.model.parameters.max_bp_trans - self.model.parameters.min_bp_trans) / 4
                        shape_bp_trans = (self.model.parameters.mean_bp_trans / sd_bp_trans) ** 2
                        scale_bp_trans = (sd_bp_trans ** 2) / self.model.parameters.mean_bp_trans
                        bp_trans = random_gamma(shape_bp_trans, scale_bp_trans)
                        
                        # force of infection from movement of farmers from one pen to another
                        if herds[source].total_herd > 0 and dest != 16 and dest != 17: # dressing rooms excluded
                            # check if movement must be made if source is infected
                            # rand = 1
                            #if herds[source].total_I > 0:
                            #    rand = 0
                            #trans_I_from_source = herds[source].total_I * bp_trans / self.model.parameters.HEV_length
                            # duration normalize between 0 and 1; divided by total number of minutes in a week
                            dur = dur / 10080
                            
                            # BIOSECURITY MEASURE: avoid risky movements: fattening -> gestation, nursery or farrowing
                            if (4 <= source <= 15) and (1 <= dest <= 3) and self.model.parameters.biosec_remove_risky_move == 1:
                                dur = 0 #if np.random.random() <= 0.5 else dur
                            # other risky movements defined by UGent and ADA
                            elif ((source == 1 and dest == 2) or (source == 3 and 1 <= dest <= 2)) and self.model.parameters.biosec_remove_risky_move == 1:
                                dur = 0 #if np.random.random() <= 0.5 else dur
                            # risky movement from nursery area only
                            #if (source == 3 and 1 <= dest <= 2) and self.model.parameters.biosec_remove_risky_move == 1:
                            #    dur = 0
                            # risky movement from gestation to farrowing only
                            #if (source == 1 and dest == 2) and self.model.parameters.biosec_remove_risky_move == 1:
                            #    dur = 0
                            # clean movements going to fattening only
                            #if (4 <= source <= 15) and self.model.parameters.biosec_remove_risky_move == 1:
                            #    dur = 0
                                
                            # BIOSECURITY MEASURE: avoid movement from infected group
                            #if herds[source].total_I > 0 and self.model.parameters.biosec_remove_risky_move == 1:
                            #    dur = 0 
                            
                            if source != 4:  
                                trans_I_from_source = dur * self.model.parameters.between_herd_trans * herds[source].total_I / herds[source].total_herd
                            else: # all movements from fattening becomes movement from all fattening pens
                                total_I = 0
                                total_herd = 0
                                for i in range (4,16):
                                    total_I += herds[i].total_I
                                    total_herd += herds[i].total_herd
                                total_herd = 1 if total_herd == 0 else total_herd
                                trans_I_from_source = dur * self.model.parameters.between_herd_trans * total_I / total_herd
                        else:
                            trans_I_from_source = 0
                        # transmission rate between groups accumulates per movement to destination
                        herds[dest].statevars.trans_btwn_pens_frm_movement += trans_I_from_source
                        # all movements to fattening becomes a movement to all fattening pens
                        if dest == 4:
                            for i in range (5,16):
                                herds[dest].statevars.trans_btwn_pens_frm_movement += trans_I_from_source
        
        # print trans rate between groups
        #for i in range(self.model.parameters.nb_herds):
        #    print(herds[i].statevars.trans_btwn_pens_frm_movement)
    
    # Determine the infectious pens
    def sample_I_from_fatteners(self):
        
        herds = self.get_populations()
        
        herds[4].statevars.nb_of_sampled_I_Jf = 0
        
        # infect fatteners initially (REMOVED, ALREADY INCLUDED IN THE YAML FILE)
        #if self.statevars.step == 0:
        #    for i in range(12):
        #        # retrieve prototype definition from the model
        #        prototype = self.model.get_prototype(name='fattening_pig_E', level='animals')
        #        infected_pig = [herds[i + 4].new_atom(custom_prototype=prototype)]
        #        herds[i + 4].add_atoms(infected_pig)
        
        # Counts a fattener herd if it is infectious; infectious if at least one pig is infected
        herds[4].statevars.nb_of_sampled_I_Jf += sum(1 for i in range(12) if herds[i + 4].total_I > 0)
           
        # infect neigboring pens depending on the number of infected animals in the pen 
        # any pair of pens can infect each other since swine flu can be transmitted via aerosol
        trans_between_pens = 0
        for i in range(4, 16):
            total_I = 0
            total_herd = 0
            portion_I = 0
            for j in range(4, 16):
                if j != i:
                    total_I += herds[j].total_I
                    total_herd += herds[j].total_herd
            if total_herd != 0:
                portion_I = total_I / total_herd
            herds[i].statevars.trans_btwn_pens_frm_movement = self.model.parameters.neighboring_herd_trans * portion_I
            #print(herds[i].statevars.trans_btwn_pens_frm_movement)
    
    # Third pathway of infection among areas in the farm
    def third_trans_pathway(self):
        herds = self.get_populations()
	# Each farm can be infected by any other areas in the farm depending on the number of infection in an area
        for i in range(5):
            total_I = 0
            total_herd = 0
            portion_I = 0
            for j in range(5):
                if i != j:
                    total_I += herds[j].total_I
                    total_herd = herds[j].total_herd
                    if total_herd != 0:
                        portion_I = total_I / total_herd 
            if i == 4:
                for i in range(4,16):
                    herds[i].statevars.trans_frm_3rd_path = self.model.parameters.third_path_trans * portion_I
            else:
                herds[i].statevars.trans_frm_3rd_path = self.model.parameters.third_path_trans * portion_I

    # external pathway of disease
    def external_pathway(self):
        # transmission from vermin
        herds = self.get_populations()
        proba_encounter = float(self.model.parameters.proba_encounter)
        proba_trans = float(self.model.parameters.proba_trans)
        proba_ext_I = float(self.model.parameters.proba_ext_I)
        proba_success_biosec_ml = float(self.model.parameters.proba_success_biosec_ml)
        proba_success_biosec_opt = float(self.model.parameters.proba_success_biosec_opt)
        proba_success_biosec_pes = float(self.model.parameters.proba_success_biosec_pes)

        for h in range(16):
            #if h == 4:
            #    h = np.random.randint(4, 16)

            susceptible = herds[h].select_atoms('health_state', 'S')

            for pig in susceptible:
                p_enc = np.random.uniform(0, proba_encounter)
                p_trans = np.random.uniform(0, proba_trans)
                p_ext_I = np.random.uniform(0, proba_ext_I)
                P1 = p_enc * p_trans * p_ext_I
                #P2_pdf = pert(proba_success_biosec_pes, proba_success_biosec_ml, proba_success_biosec_opt)
                P2 = np.random.uniform(proba_success_biosec_pes,proba_success_biosec_opt) #P2_pdf.rvs(size=1)[0]
                R_contact = P1 * (1 - P2)
                #print(R_contact)
                if np.random.random() < R_contact:
                    #current_state = pig.statevars['health_state']
                    #print(f"Changing state for pig: {pig}")
                    #print(f"Current state: {current_state} (type: {type(current_state)})")
                    #print(f"Changing to state: E")
                    
                    try:
                        pig.apply_prototype(name='infected_outside_farm', prototype='infected_outside_farm', execute_actions=True)
                        #print(f"New state: {pig.statevars['health_state']} (type: {type(pig.statevars['health_state'])})")
                    except TypeError as e:
                        print(f"TypeError occurred: {e}")
                        quit()
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        quit()
