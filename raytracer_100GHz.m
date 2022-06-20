% For reproducibility
rng(7);
% Viewing Topographic map of Chicago
% viewer = siteviewer("Buildings","chicago.osm","Basemap","topographic");

% Defining base station (transmitter) sites
%%% tx properties
tx_names = ["S Clarke", "Monroe", "Adam"];
lats = [41.878313, 41.880680, 41.879405];
lons = [-87.630772, -87.630842, -87.631010];
heights = [6, 6, 6];
freq = 100e9;
transmit_power = 10;
%%% Antenna properties
antenna_size = [64, 1];
antenna_spacing = [0.0015, 0.0015];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
%%% Creating tx
tx = txsite("Name",tx_names,"Latitude",lats,"Longitude",lons,"TransmitterFrequency",freq,"Antenna",cfgArray,"AntennaHeight",heights,"TransmitterPower",transmit_power);
% disp(tx(1,1));
% show(tx);

% Defining receiver locations
%%% Cluster 1 : Straight line street
n_recv_1 = 20000;
lat_min = 41.878381;
lat_max = 41.880677;
lon_min = -87.630803;
lon_max = -87.630818;
lats_1 = lat_min + (lat_max-lat_min)*rand(1,n_recv_1);
lons_1 = lon_min + (lon_max-lon_min)*rand(1,n_recv_1);
%%% Cluster 2 : Left Bend in middle corresponding to BS 3 (Quincy)
n_recv_2 = 10000;
lat_min = 41.879383;
lat_max = 41.879431;
lon_min = -87.630944;
lon_max = -87.630633;
lats_2 = lat_min + (lat_max-lat_min)*rand(1,n_recv_2);
lons_2 = lon_min + (lon_max-lon_min)*rand(1,n_recv_2);
%%% Concatenating all users in single array and shuffling
n_recv = n_recv_1 + n_recv_2;
lats = cat(2, lats_1, lats_2);
lons = cat(2, lons_1, lons_2);
rand_ind = randperm(n_recv);
lats = lats(rand_ind);
lons = lons(rand_ind);
%%% Saving lats, lons for ingestion by 2p4GHz data generator
temp.lats = lats;
temp.lons = lons;
save ('temp.mat', '-struct', 'temp');

names = string(1:n_recv);
height = 2;
%%Antenna properties
%TODO Change Antenna Properties
antenna_size = [1, 1];
antenna_spacing = [0.0015, 0.0015];
cfgArray = arrayConfig("Size",antenna_size,"ElementSpacing",antenna_spacing);
rx = rxsite("Name",names,"Latitude",lats,"Longitude",lons,"Antenna",cfgArray,"AntennaHeight",height);
% show (rx)

% Ray Tracing Model combined with atmospheric (gas) attenuation
rtpm = propagationModel("raytracing",Method="sbr",MaxNumReflections=0);
gas_pm = propagationModel("gas");
% fog_pm = propagationModel("fog");
rtPlusWeather = rtpm + gas_pm;
rays = raytrace(tx,rx,rtPlusWeather);

% Removing old directory and creating new one
if ~exist('./Raytracing_scenarios', 'dir')
    fprintf("Creating Raytracing Scenarios directory\n")
    mkdir './Raytracing_scenarios'
end
if exist('./Raytracing_scenarios/O1_100', 'dir')
    fprintf("Deleting old O1_100 directory\n")
    rmdir './Raytracing_scenarios/O1_100' s
end
fprintf("Creating new O1_100 directory\n")
mkdir './Raytracing_scenarios/O1_100'

% Storing data for each base station
for bs = 1:length(tx_names)
    %%% CIR File (Contains phase in degrees, ToA in seconds, and received power in dbm for each path)
    CIR = [n_recv];
    DoD = [n_recv];
    DoA = [n_recv];
    LOS = [n_recv];
    curr_paths = 0;
    curr = 1;
    for i = 1:n_recv
        fprintf('Getting data for base station %i and receiver %i\n',bs,i);
        num_paths_max = 1;
        num_paths = 0;
        CIR = [CIR i 0];
        DoD = [DoD i 0];
        DoA = [DoA i 0];
        curr = curr + 2;
        while num_paths<num_paths_max
            %         fprintf ("%d %d %d %d\n", 1, i, 1, num_paths+1);
            %         fprintf ("%f %f %f\n", rays1{1,i}(1,num_paths+1).PhaseShift, rays1{1,i}(1,num_paths+1).PropagationDelay, rays1{1,i}(1,num_paths+1).PathLoss);
            try
                %LOS Status only for first path
                if num_paths==0
                    LOS_status = rays{bs,i}(1,num_paths+1).LineOfSight;
                    LOS = [LOS LOS_status];
                end
                phase = 180 / pi * rays{bs,i}(1,num_paths+1).PhaseShift;
                ToA = rays{bs,i}(1,num_paths+1).PropagationDelay;
                recv_power = -1 * (rays{bs,i}(1,num_paths+1).PathLoss + 30);
                departure_azi = rays{bs,i}(1,num_paths+1).AngleOfDeparture(1,1);
                departure_ele = rays{bs,i}(1,num_paths+1).AngleOfDeparture(2,1);
                arrival_azi = rays{bs,i}(1,num_paths+1).AngleOfArrival(1,1);
                arrival_ele = rays{bs,i}(1,num_paths+1).AngleOfArrival(2,1);
                num_paths = num_paths + 1;
                CIR = [CIR num_paths phase ToA recv_power];
                DoD = [DoD num_paths departure_azi departure_ele 9999];
                DoA = [DoA num_paths arrival_azi arrival_ele 9999];
                %fprintf ('Happening')
            catch
                %fprintf ('Done for');
                %If broke at 0 paths then put LOS_status = -1
                if num_paths == 0
                    LOS = [LOS -1];
                end
                break
            end
        end
        CIR(1,curr) = num_paths;
        DoD(1,curr) = num_paths;
        DoA(1,curr) = num_paths;
        curr = curr + 4 * num_paths;
    end
    
    %Converting arrays to .mat files for ingestion by deepMIMO generator
    
    save(strcat('./Raytracing_scenarios/O1_100/O1_100.',int2str(bs),'.CIR.mat'), 'CIR');
    save(strcat('./Raytracing_scenarios/O1_100/O1_100.',int2str(bs),'.DoD.mat'), 'DoD');
    save(strcat('./Raytracing_scenarios/O1_100/O1_100.',int2str(bs),'.DoA.mat'), 'DoA');
    save(strcat('./Raytracing_scenarios/O1_100/O1_100.',int2str(bs),'.LoS.mat'), 'LOS');
end 