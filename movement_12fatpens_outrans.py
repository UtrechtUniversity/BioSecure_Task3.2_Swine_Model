"""
Python add-on for data preprocessing and farmer movement process in the metapopulation model.
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
# Preprocessor class for restructuring farmer movement data
#===============================================================
class FarmerMovementsReader(EmulsionPreprocessor):
    """A preprocessor class for reading the CSV that describes farmer movements
    and restructuring it into a dictionary, stored in shared simulation data.
    """

    def init_preprocessor(self):
        """Check if a valid 'farmer_movement_file' is provided in the input files.
        Raise an error if not present."""
        if self.input_files is None or 'trade_file' not in self.input_files:
            raise SemanticException("A valid 'trade_file' must be specified in the input_files section for pre-processing class {}".format(self.__class__.__name__))

    def run_preprocessor(self):
        """Load farmer movements data from the CSV file into the simulation's shared data.
        
        Expected file format: CSV with the following fields:
        - date: the date of the movement (in ISO format)
        - source: ID of the source farm
        - dest: ID of the destination farm
        - age: age group of the animals being moved
        - quantity: number of animals being moved
        """
        if 'moves' not in self.simulation.shared_data:
            debuginfo('Reading {}'.format(self.input_files.trade_file))
            self.simulation.shared_data['moves'] = self.restructure_moves()
        else:
            debuginfo('Farmer movements already loaded in the simulation')

    def restructure_moves(self):
        """Restructure the farmer movement CSV data into nested dictionaries.

        The file contains columns: date, source farm, destination farm, age group, quantity.

        This method restructures the data into a dictionary format based on simulation steps:
        {
            step: {
                source_id: [(dest_id, duration), ...],
                ...
            },
            ...
        }

        - Steps are calculated based on the simulation origin_date and step_duration.
        """
        origin = self.model.origin_date
        step_duration = self.model.step_duration
        moves = {}
        with open(self.input_files.trade_file) as csvfile:
            # Read the CSV file for farmer movements
            csvreader = csv.DictReader(csvfile, delimiter=',')
            for row in csvreader:
                week = dup.parse(row['date'])
                if week < origin:
                    # Ignore movements that occurred before the simulation's start date
                    continue
                # Convert the movement date into a simulation step
                step = (week - origin) // step_duration
                # Group movements by step and source farm
                if step not in moves:
                    moves[step] = {}
                src, dest, dur = int(row['source']), int(row['dest']), float(row['duration'])
                if src not in moves[step]:
                    moves[step][src] = []
                # Append destination farm and movement duration to the list of movements for this step and source
                moves[step][src].append([dest, dur])
        return moves

#===============================================================
# CLASS Metapopulation (LEVEL 'metapop')
#===============================================================
class Metapopulation(MetapopProcessManager):
    """
    Class representing the metapopulation level.
    """

    #----------------------------------------------------------------
    # Animal and farmer movements Processes
    #----------------------------------------------------------------
    def animal_farmer_movement(self):
        """
        Manage the movement of animals and farmers between herds.

        Herd assignments:
        - herd 0: non-gestating gilts and sows (A)
        - herd 1: gestating gilts and sows (G)
        - herd 2: farrowing pigs and newborn piglets (Jnb)
        - herd 3: weaned piglets in the nursery (Jn)
        - herds 4-15: growing/fattening pigs (Jf)
        - herd 16: dressing room 1 (no pigs)
        - herd 17: dressing room 2 (no pigs)
        """
        
        # Get farmer movements and herd populations
        moves = self.simulation.shared_data['moves']
        herds = self.get_populations()
        
        ## TRANSFER PIGS TO THEIR ASSIGNED GROUP
        
        # Record gestating sows/gilts from non-gestating group
        gestating_sows = herds.select_atoms('age_group', 'G')
        # Remove gestating sows/gilts from non-gestating group
        herds.remove_atoms(gestating_sows)
        # Add gestating sows/gilts to gestating group
        herds.add_atoms(gestating_sows)
        
        # Record farrowing sows from gestating group
        farrowing_sows = herds.select_atoms('age_group', 'F')
        # Record number of farrowing sows to determine number of newborns
        herds.statevars.nb_new_farrowing_sows = len(farrowing_sows)
        # Remove farrowing sows from gestating group
        herds.remove_atoms(farrowing_sows)
        # Add farrowing sows to farrowing group
        herds.add_atoms(farrowing_sows)
        
        # Vertical transmission parameter
        vert_trans = self.model.parameters.vert_trans
        ## Produce newborns
        newborn = []
        qty = self.model.parameters.pba
        for sow in farrowing_sows:
            # Number of newborns must be less than capacity
            if herds.total_Jnb < self.model.parameters.K_herd:
                # Produced infected newborns
                if (sow.is_in_state('I') or sow.is_in_state('E')) and vert_trans > random.random():
                    newborn_prototype = 'newborn_E'
                # Susceptible sows produce susceptible newborns
                elif sow.is_in_state('S'):
                    newborn_prototype = 'newborn_S'
                # Produce newborns with maternal immunity
                else:
                    newborn_prototype = 'newborn_M'
                mean_pba = round(self.model.parameters.mean_pba)
                sd_pba = round(self.model.parameters.sd_pba)
                pba = round(random.gauss(mean_pba, sd_pba))
                newborn = newborn + [herds.new_atom(sublevel='animals', prototype=newborn_prototype) for _ in range(pba)]
                
        if len(newborn) != 0:
            herds.add_atoms(newborn)
        
        ## Transfer corresponding age group of pigs to the correct herd
# Transfer non-gestating sows to the correct herd
nongestating_sows = herds.select_atoms('age_group', 'A')
herds.remove_atoms(nongestating_sows)
herds.add_atoms(nongestating_sows)

# Transfer nursery pigs to the correct herd
nursery = herds.select_atoms('age_group', 'Jn')
herds.remove_atoms(nursery)
herds.add_atoms(nursery)

# Transfer growers to the correct herd
growers = herds.select_atoms('age_group', 'Jf')
herds.remove_atoms(growers)

## Random testing for infectious or exposed animals and then remove them from the population
# Remove infectious growers based on probability
growers = [pig for pig in growers if not (pig.is_in_state('I') and np.random.rand() < self.model.parameters.proba_removal_if_I)]
# Remove exposed growers based on probability
growers = [pig for pig in growers if not (pig.is_in_state('E') and np.random.rand() < self.model.parameters.proba_removal_if_E)]

# Calculate the excess number of growers
excess = len(growers) - 144       

## Only 12 fattening pigs per pen (12 pens), others are sold
new_gilts = []
if excess > 0:
    # Select excess growers randomly
    excess_growers = random.sample(growers, excess)
    # Some fatteners are bred to gilts
    total_sows = herds.total_A + herds.total_G + herds.total_F
    if total_sows < 3 * self.model.parameters.K_sows:  # Multiplied by 3 to get total population of sows and gilts
        new_gilts = [pig.clone(prototype='nongestating') for pig in excess_growers if pig.is_in_state('Female')]
    # Only 144 fatteners allowed
    growers = [x for x in growers if x not in excess_growers]

# Add new gilts to non-gestating group
if len(new_gilts) != 0:
    herds.add_atoms(new_gilts)

if len(growers) > 0:
    # Specify the number of parts (n)
    n = np.floor(len(growers) / 12)
    if n == 0:
        n = 1
    elif n > 12:
        n = 12
else:
    n = 1

# Use array_split to split the array into n parts
pens = np.array_split(growers, n)

# Transfer pigs to the fattening pens
for i, pen in enumerate(pens):
    herds[i + 4].add_atoms(pen)

## MOVEMENT OF FARMERS

# Reset transmission rate between groups per iteration
for i in range(self.model.parameters.nb_herds):
    herds[i].statevars.trans_btwn_pens_frm_movement = 0

if self.statevars.step in moves:
    trans_I_from_source = 0
    for source in moves[self.statevars.step]:
        for dest, dur in moves[self.statevars.step][source]:
            if source != dest: 
                # Get parameters for between pen transmission
                sd_bp_trans = (self.model.parameters.max_bp_trans - self.model.parameters.min_bp_trans) / 4
                shape_bp_trans = (self.model.parameters.mean_bp_trans / sd_bp_trans) ** 2
                scale_bp_trans = (sd_bp_trans ** 2) / self.model.parameters.mean_bp_trans
                bp_trans = random_gamma(shape_bp_trans, scale_bp_trans)
                
                ## Force of infection from movement of farmers from one pen to another
                if herds[source].total_herd > 0 and dest != 16 and dest != 17:  # Dressing rooms excluded
                    # Normalize duration between 0 and 1; divided by total number of minutes in a week
                    dur = dur / 10080
                    
                    # BIOSECURITY MEASURE: avoid risky movements: fattening -> gestation, nursery or farrowing
                    if (4 <= source <= 15) and (1 <= dest <= 3) and self.model.parameters.biosec_remove_risky_move == 1:
                        dur = 0
                    # Other risky movements defined by UGent and ADA
                    elif ((source == 1 and dest == 2) or (source == 3 and 1 <= dest <= 2)) and self.model.parameters.biosec_remove_risky_move == 1:
                        dur = 0
                        
                    # BIOSECURITY MEASURE: avoid movement from infected group
                    if source != 4:  # For non-fattening groups
                        trans_I_from_source = dur * self.model.parameters.between_herd_trans * herds[source].total_I / herds[source].total_herd
                    else:  # All movements from fattening become movement from all fattening pens
                        total_I = 0
                        total_herd = 0
                        for i in range(4, 16):
                            total_I += herds[i].total_I
                            total_herd += herds[i].total_herd
                        total_herd = 1 if total_herd == 0 else total_herd
                        trans_I_from_source = dur * self.model.parameters.between_herd_trans * total_I / total_herd
                else:
                    trans_I_from_source = 0
                # Accumulate transmission rate between groups per movement to destination
                herds[dest].statevars.trans_btwn_pens_frm_movement += trans_I_from_source
                # All movements to fattening become a movement to all fattening pens
                if dest == 4:
                    for i in range(5, 16):
                        herds[dest].statevars.trans_btwn_pens_frm_movement += trans_I_from_source
        
        # print trans rate between groups
        #for i in range(self.model.parameters.nb_herds):
        #    print(herds[i].statevars.trans_btwn_pens_frm_movement)
    
    ## DETERMINE INFECTIOUS PENS
def sample_I_from_fatteners(self):
    """Sample infectious pens from the fatteners."""
    herds = self.get_populations()
    
    # Initialize the count of sampled infectious fatteners
    herds.statevars.nb_of_sampled_I_Jf = 0
    
    # Count a fattener herd as infectious if at least one pig is infected
    herds.statevars.nb_of_sampled_I_Jf += sum(1 for i in range(12) if herds[i + 4].total_I > 0)
       
    # Infect neighboring pens depending on the number of infected animals in the pen
    # Pen configuration is 4, 6, 8, 10, 12, 14, 16 together in a line (in this order)
    # 5, 7, 9, 11, 13, 15 together in a line (in this order)
    trans_between_pens = 0
    for i in range(4, 16):
        if i == 4:
            trans_between_pens += self.model.parameters.neighboring_herd_trans * herds[i + 2].total_I / herds[i + 2].total_herd
            herds[i].statevars.trans_btwn_pens_frm_movement += trans_between_pens
        elif i == 5:
            trans_between_pens += self.model.parameters.neighboring_herd_trans * herds[i + 2].total_I / herds[i + 2].total_herd
            herds[i].statevars.trans_btwn_pens_frm_movement += trans_between_pens
        elif i == 14:
            trans_between_pens += self.model.parameters.neighboring_herd_trans * herds[i - 2].total_I / herds[i - 2].total_herd
            herds[i].statevars.trans_btwn_pens_frm_movement += trans_between_pens
        elif i == 15:
            trans_between_pens += self.model.parameters.neighboring_herd_trans * herds[i - 2].total_I / herds[i - 2].total_herd
            herds[i].statevars.trans_btwn_pens_frm_movement += trans_between_pens
        else:
            trans_between_pens += self.model.parameters.neighboring_herd_trans * (herds[i - 2].total_I + herds[i + 2].total_I) / (herds[i - 2].total_herd + herds[i + 2].total_herd)
            herds[i].statevars.trans_btwn_pens_frm_movement += trans_between_pens

## DISEASE TRANSMISSION FROM OUTSIDE THE FARM
def external_pathway(self):
    """Simulate disease transmission from outside the farm."""
    # Get herd data
    herds = self.get_populations()
    
    # Get parameter values needed for the simulation
    proba_encounter = float(self.model.parameters.proba_encounter)
    proba_trans = float(self.model.parameters.proba_trans)
    proba_ext_I = float(self.model.parameters.proba_ext_I)
    proba_success_biosec_ml = float(self.model.parameters.proba_success_biosec_ml)
    proba_success_biosec_opt = float(self.model.parameters.proba_success_biosec_opt)
    proba_success_biosec_pes = float(self.model.parameters.proba_success_biosec_pes)

    for h in range(16):  # Loop through all groups
        # Select susceptible pigs
        susceptible = herds[h].select_atoms('health_state', 'S')

        # Check if each susceptible pig gets infected
        for pig in susceptible:  # Loop through all susceptible pigs
            # Generate probabilities related to disease transmission from outside
            p_enc = np.random.uniform(0, proba_encounter)
            p_trans = np.random.uniform(0, proba_trans)
            p_ext_I = np.random.uniform(0, proba_ext_I)
            P1 = p_enc * p_trans * p_ext_I  # Probability of transmission without biosecurity
            # Generate the probability of biosecurity success
            P2_pdf = pert(proba_success_biosec_pes, proba_success_biosec_ml, proba_success_biosec_opt)
            P2 = P2_pdf.rvs(size=1)
            # Compute the probability of transmission from an outside source for current susceptible pig
            R_contact = P1 * (1 - P2)

            if np.random.random() < R_contact:  # Infect susceptible pig if true
                try:
                    # Change the state of the pig from susceptible to exposed
                    pig.apply_prototype(name='infected_outside_farm', prototype='infected_outside_farm', execute_actions=True)
                except TypeError as e:
                    print(f"TypeError occurred: {e}")
                    quit()
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    quit()
